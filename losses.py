# losses.py
import torch
import torch.nn.functional as F
from utils import cosine_similarity

class BiLevelContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, lambda_weight=0.8):
        super().__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight # lambda in Eq 7

    def _single_contrastive_loss(self, z_anchor, z_positive, all_z):
        """ Helper for standard contrastive loss (InfoNCE variant) """
        # z_anchor, z_positive: [B, D]
        # all_z: [M, D] where M >= B (e.g., all features in the batch)
        sim_positive = cosine_similarity(z_anchor, z_positive, dim=1) / self.temperature # [B]
        sim_all = torch.mm(z_anchor, all_z.t().contiguous()) / self.temperature # [B, M]

        # Mask out positive pairs from the denominator implicitly via logsumexp? No, need explicit mask.
        # Assume all_z contains the anchors and positives. Need to mask self and positive.
        # This needs careful batch/indexing. Simpler: assume negatives are just other items in batch.

        # Simple version: negatives are other items in the same batch
        batch_size = z_anchor.shape[0]
        sim_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature # [B, B]
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_anchor.device)
        sim_matrix.masked_fill_(mask, -float('inf')) # Mask self

        # log( exp(pos) / sum(exp(neg)) ) = pos - logsumexp(neg)
        # Here, pos is diagonal of sim(anchor, positive_batch) - already computed
        log_sum_exp_neg = torch.logsumexp(sim_matrix, dim=1) # LogSumExp over negatives for each anchor

        loss = -torch.mean(sim_positive - log_sum_exp_neg) # Average over batch
        return loss

    def forward(self, zI, zP, zB, zF):
        # zI, zP, zB, zF are features projected into the SAME latent space [B, D]
        batch_size = zI.shape[0]
        if batch_size <= 1: return torch.tensor(0.0, device=zI.device) # Need pairs for loss

        # --- Tumor-level Contrastive Losses (Eq 4, 5) ---
        # L_I2P: Image as anchor, Points as positive
        loss_i2p = self._single_contrastive_loss(zI, zP, zP) # Negatives are other points features

        # L_P2I: Points as anchor, Image as positive
        loss_p2i = self._single_contrastive_loss(zP, zI, zI) # Negatives are other image features

        # Average tumor-level loss
        loss_tumor = (loss_i2p + loss_p2i) / 2.0

        # --- Brain-level Contrastive Loss (Eq 6) ---
        # L_B2F: Brain network as anchor, Focal tumor as positive
        loss_b2f = self._single_contrastive_loss(zB, zF, zF) # Negatives are other focal features

        # --- Combine Losses (Eq 7) ---
        total_loss = self.lambda_weight * loss_tumor + (1.0 - self.lambda_weight) * loss_b2f
        # Lmulti = total_loss / batch_size # Paper divides by M (batch size) outside log

        return total_loss