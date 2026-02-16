import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectAwareContrastiveLoss(nn.Module):
    """
    Subject-Aware Contrastive Loss (SupCon-like).
    features: (2B, D)  labels: (B,)
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = labels.shape[0]

        labels = torch.cat([labels, labels], dim=0)  # (2B,)
        features = F.normalize(features, dim=1)

        sim = torch.matmul(features, features.T) / self.temperature  # (2B,2B)

        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = (-mean_log_prob_pos).mean()

        with torch.no_grad():
            sim_for_acc = sim.clone()
            sim_for_acc.fill_diagonal_(-1e9)
            pred_ids = torch.argmax(sim_for_acc, dim=1)
            correct = torch.eq(labels[pred_ids], labels).float()
            acc = correct.mean()

        return loss, acc


class CrossModalClipLoss(nn.Module):
    """
    CLIP / InfoNCE 跨模态对齐：
    给定一批样本 i=1..B，
      ecg_i 与 ppg_i 是正样本对
      ecg_i 与 ppg_j (j!=i) 是负样本
    返回 loss 和 top1 acc（检索准确率）
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, ecg_feat, ppg_feat):
        """
        ecg_feat: (B, D)
        ppg_feat: (B, D)
        """
        ecg_feat = F.normalize(ecg_feat, dim=1)
        ppg_feat = F.normalize(ppg_feat, dim=1)

        logits = (ecg_feat @ ppg_feat.t()) / self.temperature  # (B,B)
        targets = torch.arange(logits.size(0), device=logits.device)

        loss_e2p = F.cross_entropy(logits, targets)
        loss_p2e = F.cross_entropy(logits.t(), targets)
        loss = (loss_e2p + loss_p2e) * 0.5

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            acc = (pred == targets).float().mean()

        return loss, acc
