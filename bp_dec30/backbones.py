#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型架构定义 (Model Architectures)
包含 ECG/PPG 编码器骨干网络和 AgeModel 微调模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== PPGFounder 骨干网络 =====================
class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.kernel_size = kernel_size; self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
    def forward(self, x):
        L = x.shape[-1]; out_dim = (L + self.stride - 1)//self.stride
        p = max(0, (out_dim - 1)*self.stride + self.kernel_size - L)
        left = p//2; right = p - left
        return self.conv(F.pad(x, (left, right)))

class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size; self.pool = nn.MaxPool1d(kernel_size)
    def forward(self, x):
        p = max(0, self.kernel_size-1); left = p//2; right = p - left
        return self.pool(F.pad(x, (left, right)))

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _ = x.size()
        avg = self.mlp(self.avg_pool(x).view(b,c))
        mx  = self.mlp(self.max_pool(x).view(b,c))
        w = self.sigmoid(avg + mx).view(b,c,1)
        return x * w

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size-1)//2
        self.conv = nn.Conv1d(2,1,kernel_size=kernel_size,padding=pad,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg,mx], dim=1)))
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
        self.use_bn, self.use_do, self.downsample = use_bn, use_do, downsample
        mid = int(out_channels*ratio)
        if use_bn: self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = Swish(); self.do1 = nn.Dropout(0.5)
        self.conv1 = MyConv1dPadSame(in_channels, mid, 1, 1, groups=1)
        if use_bn: self.bn2 = nn.BatchNorm1d(mid)
        self.act2 = Swish(); self.do2 = nn.Dropout(0.5)
        self.conv2 = MyConv1dPadSame(mid, mid, kernel_size, stride if downsample else 1, groups=groups)
        if use_bn: self.bn3 = nn.BatchNorm1d(mid)
        self.act3 = Swish(); self.do3 = nn.Dropout(0.5)
        self.conv3 = MyConv1dPadSame(mid, out_channels, 1, 1, groups=1)
        r=2; self.se1 = nn.Linear(out_channels, out_channels//r)
        self.se2 = nn.Linear(out_channels//r, out_channels)
        self.is_first_block = is_first_block
        if self.downsample: self.pool = MyMaxPool1dPadSame(kernel_size=stride)
        self.in_ch, self.out_ch = in_channels, out_channels
    def forward(self, x):
        idt = x; out = x
        if not self.is_first_block:
            if self.use_bn: out = self.bn1(out)
            out = self.act1(out); 
            if self.use_do: out = self.do1(out)
        out = self.conv1(out)
        if self.use_bn: out = self.bn2(out)
        out = self.act2(out); 
        if self.use_do: out = self.do2(out)
        out = self.conv2(out)
        if self.use_bn: out = self.bn3(out)
        out = self.act3(out); 
        if self.use_do: out = self.do3(out)
        out = self.conv3(out)
        se = torch.mean(out, dim=-1)
        se = torch.sigmoid(self.se2(Swish()(self.se1(se))))
        out = torch.einsum("bct,bc->bct", out, se)
        if self.downsample: idt = self.pool(idt)
        if self.out_ch != self.in_ch:
            idt = idt.transpose(-1,-2)
            ch1 = (self.out_ch-self.in_ch)//2; ch2 = self.out_ch-self.in_ch-ch1
            idt = F.pad(idt, (ch1,ch2)); idt = idt.transpose(-1,-2)
        return out + idt

class StagePPG(nn.Module):
    def __init__(self, in_ch, out_ch, ratio, kernel, stride, groups_width, i_stage, m_blocks, use_bn=True, use_do=True):
        super().__init__()
        blocks=[]
        for i in range(m_blocks):
            down = (i==0); st = stride if i==0 else 1
            is_first = (i_stage==0 and i==0)
            blocks.append(BasicBlockPPG(in_ch if i==0 else out_ch, out_ch, ratio, kernel, st, out_ch//groups_width, down, is_first, use_bn, use_do))
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x):
        for b in self.blocks: x=b(x)
        return x

class Net1D_PPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = MyConv1dPadSame(1, 32, 3, 1)
        self.first_bn = nn.BatchNorm1d(32); self.first_act = Swish()
        filt=[32,64,128,256,512]; blocks=[3,4,4,4,2]
        stages=[]; in_ch=32
        for i,(oc,mb) in enumerate(zip(filt,blocks)):
            stages.append(StagePPG(in_ch, oc, 1, 3, 2, 16, i, mb, True, True)); in_ch=oc
        self.stages=nn.ModuleList(stages)
    def forward(self,x):
        x=self.first_act(self.first_bn(self.first(x)))
        for s in self.stages: x=s(x)
        return x  # (B,512,T')

class PPGEncoderCLIP(nn.Module):
    """PPG path used in pretrain: backbone -> CBAM -> mean-pool -> 512 -> proj(512->512)"""
    def __init__(self, with_proj=True, proj_hidden=None):
        super().__init__()
        self.backbone = Net1D_PPG()
        self.cbam = CBAM1D(in_channels=512)
        self.with_proj = with_proj
        if with_proj:
            if proj_hidden and proj_hidden>0:
                self.proj = nn.Sequential(
                    nn.Linear(512, proj_hidden),
                    nn.GELU(),
                    nn.Linear(proj_hidden, 512)
                )
            else:
                self.proj = nn.Linear(512,512)
        else:
            self.proj = nn.Identity()
    def forward(self, x):  # x: (B,1,7500)
        feat = self.backbone(x)
        feat = self.cbam(feat)
        z = feat.mean(dim=-1)  # (B,512)
        z = self.proj(z)
        return z

# ===================== ECGFounder 1-lead 骨干网络 =====================
class MyConv1dPadSame_ECG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.kernel_size=kernel_size; self.stride=stride
        self.conv=nn.Conv1d(in_channels,out_channels,kernel_size,stride,groups=groups)
    def forward(self,x):
        L=x.shape[-1]; out=(L+self.stride-1)//self.stride
        p=max(0,(out-1)*self.stride+self.kernel_size-L)
        l=p//2; r=p-l
        return self.conv(F.pad(x,(l,r)))

class MyMaxPool1dPadSame_ECG(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k=k; self.pool=nn.MaxPool1d(k)
    def forward(self,x):
        p=max(0,self.k-1); l=p//2; r=p-l
        return self.pool(F.pad(x,(l,r)))

class SwishECG(nn.Module):
    def forward(self,x): return x*torch.sigmoid(x)

class BasicBlockECG(nn.Module):
    def __init__(self,in_ch,out_ch,ratio,ksize,stride,groups,downsample,
                 is_first_block=False,use_bn=False,use_do=False):
        super().__init__()
        mid=int(out_ch*ratio)
        self.bn1=nn.BatchNorm1d(in_ch); self.act1=SwishECG(); self.do1=nn.Dropout(0.5)
        self.conv1=MyConv1dPadSame_ECG(in_ch,mid,1,1,groups=1)
        self.bn2=nn.BatchNorm1d(mid); self.act2=SwishECG(); self.do2=nn.Dropout(0.5)
        self.conv2=MyConv1dPadSame_ECG(mid,mid,ksize,stride if downsample else 1,groups=groups)
        self.bn3=nn.BatchNorm1d(mid); self.act3=SwishECG(); self.do3=nn.Dropout(0.5)
        self.conv3=MyConv1dPadSame_ECG(mid,out_ch,1,1,groups=1)
        r=2; self.se1=nn.Linear(out_ch,out_ch//r); self.se2=nn.Linear(out_ch//r,out_ch)
        self.downsample=downsample
        if self.downsample: self.pool=MyMaxPool1dPadSame_ECG(stride)
        self.in_ch=in_ch; self.out_ch=out_ch; self.is_first_block=is_first_block
    def forward(self,x):
        idt=x; out=x
        if not self.is_first_block:
            out=self.bn1(out); out=self.act1(out); out=self.do1(out)
        out=self.conv1(out)
        out=self.bn2(out); out=self.act2(out); out=self.do2(out); out=self.conv2(out)
        out=self.bn3(out); out=self.act3(out); out=self.do3(out); out=self.conv3(out)
        se=out.mean(-1); se=self.se2(self.act3(self.se1(se))); se=torch.sigmoid(se)
        out=torch.einsum("bcl,bc->bcl", out, se)
        if self.downsample: idt=self.pool(idt)
        if self.out_ch!=self.in_ch:
            idt=idt.transpose(-1,-2); ch1=(self.out_ch-self.in_ch)//2; ch2=self.out_ch-self.in_ch-ch1
            idt=F.pad(idt,(ch1,ch2)); idt=idt.transpose(-1,-2)
        return out+idt

class StageECG(nn.Module):
    def __init__(self,in_ch,out_ch,ratio,ksize,stride,groups_width,i_stage,m_blocks,use_bn=False,use_do=False):
        super().__init__()
        blocks=[]
        for i in range(m_blocks):
            down=(i==0); blk=BasicBlockECG(in_ch if i==0 else out_ch,out_ch,ratio,ksize,stride,out_ch//groups_width,down,is_first_block=(i_stage==0 and i==0),use_bn=use_bn,use_do=use_do)
            blocks.append(blk)
        self.blocks=nn.ModuleList(blocks)
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return x

class ECGBackbone1L(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=MyConv1dPadSame_ECG(1,64,16,2)
        self.first_bn=nn.BatchNorm1d(64); self.first_act=SwishECG()
        filt=[64,160,160,400,400,1024,1024]; m=[2,2,2,3,3,4,4]
        stages=[]; in_ch=64
        for i,(oc,mb) in enumerate(zip(filt,m)):
            stages.append(StageECG(in_ch,oc,1,16,2,16,i,mb,False,False)); in_ch=oc
        self.stages=nn.ModuleList(stages)
    def forward(self,x):
        x=self.first(x); x=self.first_bn(x); x=self.first_act(x)
        for s in self.stages: x=s(x)
        return x # (B,1024,L_enc~30)

class ECGEncoderCLIP(nn.Module):
    """ECG path used in pretrain: backbone -> mean-pool -> 1024 -> proj(1024->512)"""
    def __init__(self, with_proj=True, proj_hidden=None):
        super().__init__()
        self.backbone = ECGBackbone1L()
        self.with_proj = with_proj
        if with_proj:
            if proj_hidden and proj_hidden>0:
                self.proj = nn.Sequential(
                    nn.Linear(1024, proj_hidden),
                    nn.GELU(),
                    nn.Linear(proj_hidden, 512)
                )
            else:
                self.proj = nn.Linear(1024, 512)
        else:
            self.proj = nn.Identity()
    def forward(self,x): # (B,1,7500)
        feat=self.backbone(x)      # (B,1024,L)
        pooled=feat.mean(dim=-1)   # (B,1024)
        z=self.proj(pooled)        # (B,512)
        return z

# ===================== Age-only Finetune Model =====================
class AgeModel(nn.Module):
    def __init__(self, modality="both", proj_hidden=0):
        super().__init__()
        assert modality in ["ppg","ecg","both"]
        self.modality = modality
        self.ppg_enc = PPGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)
        self.ecg_enc = ECGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)
        in_dim = 512 if modality!="both" else 1024
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def load_from_pretrain(self, ckpt_path: str, device):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        own = self.state_dict()
        loadable = {k:v for k,v in state.items() if k in own and ("ecg_enc" in k or "ppg_enc" in k)}
        msg = self.load_state_dict({**own, **loadable}, strict=False)
        print(f"[load pretrain] from {ckpt_path}\n  missing={msg.missing_keys}\n  unexpected={msg.unexpected_keys}")
        self.to(device)
        
    def forward(self, ecg, ppg):
        zs=[]
        if self.modality in ["ecg","both"]:
            z_ecg = self.ecg_enc(ecg)    # (B,512)
            zs.append(z_ecg)
        if self.modality in ["ppg","both"]:
            z_ppg = self.ppg_enc(ppg)    # (B,512)
            zs.append(z_ppg)
        z = zs[0] if len(zs)==1 else torch.cat(zs, dim=1)  # (B, 512/1024)
        raw = self.head(z).squeeze(1)     # (B,)
        return raw