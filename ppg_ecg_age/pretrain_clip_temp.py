#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, json, argparse, random, datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ---------------------------
# 默认路径（按你的要求）
# ---------------------------
DEF_NPZ_DIR = "/home/notebook/data/personal/S9061270/age_ready/unlabeled_zscore"
DEF_PPG_CKPT = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/best_checkpoint.pth"
DEF_ECG_CKPT = "/home/notebook/data/personal/S9061270/model/ecg_founder/checkpoint/1_lead_ECGFounder.pth"
DEF_OUT_DIR  = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/pretrain_temp"

# ---------------------------
# 复用：PPG & ECG 编码器（与你已跑通的版本一致，结构保持不变）
# ---------------------------

class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
    def forward(self, x):
        L = x.shape[-1]
        out_dim = (L + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - L)
        left = p // 2; right = p - left
        return self.conv(F.pad(x, (left, right)))

class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)
    def forward(self, x):
        p = max(0, self.kernel_size - 1)
        left = p // 2; right = p - left
        return self.pool(F.pad(x, (left, right)))

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        att = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * att

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * att

class CBAM1D(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(in_channels, reduction)
        self.sa = SpatialAttention1D(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

class BasicBlockPPG(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, downsample,
                 is_first_block=False, use_bn=True, use_do=True):
        super().__init__()
        mid = int(out_channels * ratio)
        self.use_bn, self.use_do, self.downsample = use_bn, use_do, downsample
        if use_bn: self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = Swish(); self.do1 = nn.Dropout(0.5)
        self.conv1 = MyConv1dPadSame(in_channels, mid, 1, 1)
        if use_bn: self.bn2 = nn.BatchNorm1d(mid)
        self.act2 = Swish(); self.do2 = nn.Dropout(0.5)
        self.conv2 = MyConv1dPadSame(mid, mid, kernel_size, stride if downsample else 1, groups=groups)
        if use_bn: self.bn3 = nn.BatchNorm1d(mid)
        self.act3 = Swish(); self.do3 = nn.Dropout(0.5)
        self.conv3 = MyConv1dPadSame(mid, out_channels, 1, 1)
        r = 2
        self.se_fc1 = nn.Linear(out_channels, out_channels // r)
        self.se_fc2 = nn.Linear(out_channels // r, out_channels)
        self.is_first_block = is_first_block
        if downsample: self.pool = MyMaxPool1dPadSame(stride)
        self.in_ch, self.out_ch = in_channels, out_channels
    def forward(self, x):
        idt = x; out = x
        if self.use_bn: out = self.bn1(out)
        out = self.act1(out)
        if self.use_do: out = self.do1(out)
        out = self.conv1(out)
        if self.use_bn: out = self.bn2(out)
        out = self.act2(out)
        if self.use_do: out = self.do2(out)
        out = self.conv2(out)
        if self.use_bn: out = self.bn3(out)
        out = self.act3(out)
        if self.use_do: out = self.do3(out)
        out = self.conv3(out)
        se = torch.mean(out, dim=-1)
        se = self.se_fc2(Swish()(self.se_fc1(se)))
        se = torch.sigmoid(se)
        out = torch.einsum("bct,bc->bct", out, se)
        if self.downsample: idt = self.pool(idt)
        if self.out_ch != self.in_ch:
            idt = idt.transpose(-1, -2)
            ch1 = (self.out_ch - self.in_ch) // 2
            ch2 = self.out_ch - self.in_ch - ch1
            idt = F.pad(idt, (ch1, ch2))
            idt = idt.transpose(-1, -2)
        return out + idt

class BasicStagePPG(nn.Module):
    def __init__(self, in_ch, out_ch, ratio, k, stride, groups, i_stage, m_blocks, use_bn=True, use_do=True):
        super().__init__()
        blocks = []
        for i in range(m_blocks):
            down = (i == 0)
            st = stride if down else 1
            blocks.append(BasicBlockPPG(in_ch if i == 0 else out_ch, out_ch, ratio, k, st,
                                        groups=out_ch // 16, downsample=down, use_bn=use_bn, use_do=use_do))
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x

class PPGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = MyConv1dPadSame(1, 32, 3, 1)
        self.first_bn = nn.BatchNorm1d(32)
        self.first_act = Swish()
        filt = [32, 64, 128, 256, 512]
        mblk = [3, 4, 4, 4, 2]
        stages = []
        in_ch = 32
        for i, (oc, m) in enumerate(zip(filt, mblk)):
            stages.append(BasicStagePPG(in_ch, oc, ratio=1, k=3, stride=2, groups=16, i_stage=i, m_blocks=m))
            in_ch = oc
        self.stages = nn.ModuleList(stages)
        self.cbam = CBAM1D(512)
    def forward(self, x):  # (B,1,7500) -> (B,512,T)
        x = self.first_act(self.first_bn(self.first(x)))
        for s in self.stages: x = s(x)
        x = self.cbam(x)
        return x  # (B,512,T)

class PPGEncoder(nn.Module):
    def __init__(self, ppg_ckpt: str):
        super().__init__()
        self.backbone = PPGBackbone()
        ckpt = torch.load(ppg_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        state = {k: v for k, v in state.items() if not k.startswith("lastlayer.")}
        msg = self.backbone.load_state_dict(state, strict=False)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(f"[PPG] loaded {ppg_ckpt} | {msg}")
        else:
            print(f"[PPG] loaded {ppg_ckpt} | {msg}")
    def forward(self, x):  # x: (B,1,7500)
        feat = self.backbone(x)           # (B,512,T)
        emb  = F.normalize(feat.mean(-1), dim=1)  # (B,512)
        return emb, feat

# ===== ECG founder =====
class MyConv1dPadSameE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, groups=groups)
    def forward(self, x):
        L = x.shape[-1]
        out_len = (L + self.stride - 1) // self.stride
        p = max(0, (out_len - 1) * self.stride + self.kernel_size - L)
        left = p // 2; right = p - left
        return self.conv(F.pad(x, (left, right)))

class MyMaxPool1dPadSameE(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size)
    def forward(self, x):
        p = max(0, self.kernel_size - 1)
        left = p // 2; right = p - left
        return self.pool(F.pad(x, (left, right)))

class SwishE(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class BasicBlockECG(nn.Module):
    def __init__(self, in_ch, out_ch, ratio, ksize, stride, groups, downsample,
                 is_first_block=False, use_bn=False, use_do=False):
        super().__init__()
        mid = int(out_ch * ratio)
        self.use_bn, self.use_do = use_bn, use_do
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = SwishE(); self.do1 = nn.Dropout(0.5)
        self.conv1 = MyConv1dPadSameE(in_ch, mid, 1, 1)
        self.bn2 = nn.BatchNorm1d(mid)
        self.act2 = SwishE(); self.do2 = nn.Dropout(0.5)
        self.conv2 = MyConv1dPadSameE(mid, mid, ksize, stride if downsample else 1, groups=groups)
        self.bn3 = nn.BatchNorm1d(mid)
        self.act3 = SwishE(); self.do3 = nn.Dropout(0.5)
        self.conv3 = MyConv1dPadSameE(mid, out_ch, 1, 1)
        self.se1 = nn.Linear(out_ch, out_ch // 2)
        self.se2 = nn.Linear(out_ch // 2, out_ch)
        self.downsample = downsample
        if downsample: self.pool = MyMaxPool1dPadSameE(kernel_size=stride)
        self.in_ch, self.out_ch = in_ch, out_ch
    def forward(self, x):
        idt = x; out = x
        if self.use_bn: out = self.bn1(out)
        out = self.act1(out)
        if self.use_do: out = self.do1(out)
        out = self.conv1(out)
        if self.use_bn: out = self.bn2(out)
        out = self.act2(out)
        if self.use_do: out = self.do2(out)
        out = self.conv2(out)
        if self.use_bn: out = self.bn3(out)
        out = self.act3(out)
        if self.use_do: out = self.do3(out)
        out = self.conv3(out)
        se = torch.mean(out, dim=-1)
        se = self.se2(self.act3(self.se1(se)))
        se = torch.sigmoid(se)
        out = torch.einsum("bcl,bc->bcl", out, se)
        if self.downsample: idt = self.pool(idt)
        if self.out_ch != self.in_ch:
            idt = idt.transpose(-1, -2)
            ch1 = (self.out_ch - self.in_ch) // 2
            ch2 = self.out_ch - self.in_ch - ch1
            idt = F.pad(idt, (ch1, ch2))
            idt = idt.transpose(-1, -2)
        return out + idt

class BasicStageECG(nn.Module):
    def __init__(self, in_ch, out_ch, ratio, k, stride, groups, i_stage, m_blocks, use_bn=False, use_do=False):
        super().__init__()
        blocks=[]
        for i in range(m_blocks):
            down = (i == 0)
            blk = BasicBlockECG(in_ch if i==0 else out_ch, out_ch, ratio, k, stride, groups=out_ch//16,
                                downsample=down, use_bn=use_bn, use_do=use_do)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x

class ECGBackbone1L(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = MyConv1dPadSameE(1, 64, 16, stride=2)
        self.first_bn = nn.BatchNorm1d(64); self.first_act = SwishE()
        filt=[64,160,160,400,400,1024,1024]
        mblk=[2,2,2,3,3,4,4]
        stages=[]; in_ch=64
        for i,oc in enumerate(filt):
            stages.append(BasicStageECG(in_ch, oc, 1, 16, 2, groups=oc//16, i_stage=i, m_blocks=mblk[i]))
            in_ch=oc
        self.stages=nn.ModuleList(stages)
    def forward(self, x):
        x = self.first_act(self.first_bn(self.first(x)))
        for s in self.stages: x = s(x)
        return x  # (B,1024,T)

class ECGEncoder(nn.Module):
    def __init__(self, ecg_ckpt: str):
        super().__init__()
        self.backbone = ECGBackbone1L()
        ckpt = torch.load(ecg_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        new_state={}
        for k,v in state.items():
            k2 = k[7:] if k.startswith("module.") else k
            if k2.startswith("dense."): continue
            new_state[k2]=v
        miss, unexp = self.backbone.load_state_dict(new_state, strict=False)
        if dist.is_initialized():
            if dist.get_rank()==0:
                print(f"[ECG] loaded {ecg_ckpt}\n  missing={miss}\n  unexpected={unexp}")
        else:
            print(f"[ECG] loaded {ecg_ckpt}\n  missing={miss}\n  unexpected={unexp}")
        self.proj = nn.Linear(1024, 512)
    def forward(self, x):  # (B,1,7500)
        feat = self.backbone(x)           # (B,1024,T)
        pooled = feat.mean(-1)           # (B,1024)
        z = F.normalize(self.proj(pooled), dim=1)  # (B,512)
        return z, feat

# ---------------------------
# CLIP 头（支持温度参数可选训练 + 独立学习率）
# ---------------------------
class ECGPPGCLIP(nn.Module):
    def __init__(self, ppg_ckpt: str, ecg_ckpt: str, learn_temp: bool = False):
        super().__init__()
        self.ppg_enc = PPGEncoder(ppg_ckpt)
        self.ecg_enc = ECGEncoder(ecg_ckpt)
        init_logit = math.log(1/0.07)  # CLIP 常用初始化
        if learn_temp:
            self.logit_scale = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_logit, dtype=torch.float32))
        self.learn_temp = learn_temp

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        z_ppg, _ = self.ppg_enc(ppg)   # (B,512)
        z_ecg, _ = self.ecg_enc(ecg)   # (B,512)
        # 非原地裁剪，保证数值稳定：scale ∈ [1/50, 50]  => T ∈ [0.02, 50]
        scale = torch.exp(self.logit_scale)
        scale = torch.clamp(scale, 1/50, 50)
        temp = 1.0 / float(scale.detach().item())
        return z_ecg, z_ppg, scale, temp

# ---------------------------
# 数据
# ---------------------------
class ECGPPGDataset(Dataset):
    """从 npz 读 (7500,2)，列0=ECG, 列1=PPG；均已 z-score；直接 (1,7500) 返回"""
    def __init__(self, files: List[str]):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p)
        x = d["x"].astype(np.float32)  # (7500,2)
        ecg = torch.from_numpy(x[:,0:1].T.copy())  # (1,7500)
        ppg = torch.from_numpy(x[:,1:2].T.copy())  # (1,7500)
        return ecg, ppg

# ---------------------------
# 分布式与训练工具
# ---------------------------
def ddp_setup(backend="nccl"):
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=1800))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def infoNCE_logits(z_a, z_b, scale):
    # z_*: (B,512), 已 L2-Norm
    logits_ab = scale * (z_a @ z_b.t())  # (B,B)
    logits_ba = logits_ab.t()
    return logits_ab, logits_ba

def clip_loss(z_ecg, z_ppg, scale):
    B = z_ecg.size(0)
    logits_ab, logits_ba = infoNCE_logits(z_ecg, z_ppg, scale)
    gt = torch.arange(B, device=z_ecg.device)
    loss_ab = F.cross_entropy(logits_ab, gt)
    loss_ba = F.cross_entropy(logits_ba, gt)
    loss = 0.5*(loss_ab + loss_ba)
    with torch.no_grad():
        prob = 0.5*(F.softmax(logits_ab,1).diag().mean() + F.softmax(logits_ba,1).diag().mean())
    return loss, float(prob.item())

@torch.no_grad()
def run_eval_ddp(model, loader, device):
    model.eval()
    tot_loss = 0.0; tot_cnt = 0
    for ecg, ppg in loader:
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        z_e, z_p, scale, _ = model(ecg, ppg)
        loss, _ = clip_loss(z_e, z_p, scale)
        bs = ecg.size(0)
        tot_loss += float(loss.item()) * bs
        tot_cnt += bs
    # all-reduce 平均
    t = torch.tensor([tot_loss, tot_cnt], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    mean_loss = (t[0] / t[1]).item()
    return mean_loss

def save_ckpt(model_wo_ddp, opt, scaler, out_dir, epoch, best=False):
    out = Path(out_dir) / ("best.pth" if best else f"epoch_{epoch}.pth")
    state = {
        "epoch": epoch,
        "model": model_wo_ddp.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(state, str(out))
    return str(out)

# ---------------------------
# 主函数
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="ECG-PPG CLIP pretraining (DDP) with learnable temperature and separate LR.")
    ap.add_argument("--npz_dir",   default=DEF_NPZ_DIR)
    ap.add_argument("--ppg_ckpt",  default=DEF_PPG_CKPT)
    ap.add_argument("--ecg_ckpt",  default=DEF_ECG_CKPT)
    ap.add_argument("--out_dir",   default=DEF_OUT_DIR)

    ap.add_argument("--epochs", type=int, default=50)
    # 单卡（per-GPU）batch size：
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)

    # 优化器
    ap.add_argument("--lr", type=float, default=1e-4, help="主干参数学习率")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_temp", type=float, default=5e-4, help="温度参数学习率（仅 --learn_temp 时生效, wd=0）")

    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val_ratio", type=float, default=0.02)

    # 温度是否训练
    ap.add_argument("--learn_temp", action="store_true", default=True, help="训练温度参数 logit_scale；若不设，则固定温度")

    args = ap.parse_args()

    # 允许单卡直接跑（torchrun 会注入这些环境变量；单卡时补齐）
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    ddp_setup("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"DDP init: WORLD_SIZE={dist.get_world_size()} | device={device}")
        print(f"Paths: npz={args.npz_dir} | ppg_ckpt={args.ppg_ckpt} | ecg_ckpt={args.ecg_ckpt} | out={args.out_dir}")
        print(f"Hyper: lr={args.lr} wd={args.weight_decay} | learn_temp={args.learn_temp} lr_temp={args.lr_temp} | per-GPU batch={args.batch_size}")

    set_seed(2025 + rank)  # 各 rank 不同 seed

    # 划分数据
    all_files = sorted([str(p) for p in Path(args.npz_dir).glob("*.npz")])
    assert len(all_files) > 0, f"No npz under {args.npz_dir}"
    n_total = len(all_files)
    n_val = max(1, int(n_total * args.val_ratio))
    rng = np.random.default_rng(2025)
    idx = np.arange(n_total); rng.shuffle(idx)
    val_idx = set(idx[:n_val].tolist())
    tr_files = [all_files[i] for i in idx if i not in val_idx]
    va_files = [all_files[i] for i in idx if i in val_idx]

    ds_tr = ECGPPGDataset(tr_files)
    ds_va = ECGPPGDataset(va_files)

    smp_tr = DistributedSampler(ds_tr, shuffle=True, drop_last=True)
    smp_va = DistributedSampler(ds_va, shuffle=False, drop_last=False)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=smp_tr,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=(args.num_workers>0))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, sampler=smp_va,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=(args.num_workers>0))

    # 模型
    model_core = ECGPPGCLIP(args.ppg_ckpt, args.ecg_ckpt, learn_temp=args.learn_temp).to(device)
    model = DDP(model_core, device_ids=[local_rank], output_device=local_rank,
                static_graph=True, find_unused_parameters=False)

    # 优化器：主参数组 + 温度参数组（仅当 learn_temp 时）
    main_params: List[torch.nn.Parameter] = []
    temp_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("logit_scale"):
            temp_params.append(p)
        else:
            main_params.append(p)
    optim_groups = []
    if len(main_params) > 0:
        optim_groups.append({"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if len(temp_params) > 0:
        optim_groups.append({"params": temp_params, "lr": args.lr_temp, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(optim_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf"); best_ep = -1; wait = 0

    for ep in range(1, args.epochs+1):
        model.train()
        smp_tr.set_epoch(ep)
        if rank == 0:
            pbar = tqdm(dl_tr, desc=f"Train (ep {ep}/{args.epochs})", leave=False)
        else:
            pbar = dl_tr

        running = 0.0; seen = 0
        for ecg, ppg in pbar:
            ecg = ecg.to(device, non_blocking=True)
            ppg = ppg.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                z_e, z_p, scale, temp = model(ecg, ppg)
                loss, pos_prob = clip_loss(z_e, z_p, scale)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            bsz = ecg.size(0)
            running += float(loss.item()) * bsz
            seen += bsz
            if rank == 0:
                pbar.set_postfix(loss=f"{running/seen:.4f}", temp=f"{temp:.3f}", pos=f"{pos_prob:.3f}")
        if rank == 0 and hasattr(pbar, "close"): pbar.close()

        # 验证
        val_loss = run_eval_ddp(model, dl_va, device)

        if rank == 0:
            print(f"[E{ep}] train_loss={running/seen:.6f} | val_loss={val_loss:.6f}")
            improved = val_loss < best - 1e-6
            stop_now = False
            if improved:
                best = val_loss; best_ep = ep; wait = 0
                ck = save_ckpt(model.module, optimizer, scaler, out_dir, ep, best=True)
                print(f"[best] val_loss={best:.6f} @epoch{ep} | saved: {ck}")
            else:
                wait += 1
                if wait >= args.patience:
                    stop_now = True
                    print(f"[early stop] no improve for {args.patience} epochs (best @ {best_ep})")
        else:
            stop_now = False

        stop_tensor = torch.tensor([1 if stop_now else 0], device=device)
        dist.broadcast(stop_tensor, src=0)
        stop_now = bool(stop_tensor.item())
        if stop_now:
            dist.barrier()
            break

    if rank == 0:
        with open(out_dir / "train_cfg.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    cleanup()

if __name__ == "__main__":
    main()
