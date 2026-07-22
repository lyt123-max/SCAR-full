from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoReMADConfig


class STSDDecomposer(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, d_hidden: int = 64, lowpass_center: float = 0.25):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.freq_mask_logits = nn.Parameter(torch.zeros(1, n_channels, n_freq))
        with torch.no_grad():
            freqs = torch.linspace(0, 1, n_freq)
            init_val = 4.0 * (lowpass_center - freqs)
            self.freq_mask_logits.copy_(init_val.unsqueeze(0).unsqueeze(0).expand(1, n_channels, -1))

        self.correction = nn.Sequential(
            nn.Conv1d(n_channels * 2, d_hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_hidden, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, length, _ = x.shape
        x_freq = torch.fft.rfft(x, dim=1).permute(0, 2, 1)
        s0_freq = x_freq * torch.sigmoid(self.freq_mask_logits)
        s0 = torch.fft.irfft(s0_freq.permute(0, 2, 1), n=length, dim=1)
        delta = self.correction(torch.cat([x, s0], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        slow = s0 + delta
        residual = x - slow
        return slow, residual

    @staticmethod
    def smoothness_loss(slow: torch.Tensor) -> torch.Tensor:
        d2 = slow[:, 2:, :] - 2 * slow[:, 1:-1, :] + slow[:, :-2, :]
        return d2.abs().mean()


class StateEncoder(nn.Module):
    def __init__(self, n_channels: int, d_state: int = 128, d_hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, d_hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_state),
        )

    def forward(self, slow: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(slow.permute(0, 2, 1))
        pooled = self.pool(hidden).squeeze(-1)
        return self.mlp(pooled)


class ChannelModulation(nn.Module):
    def __init__(self, d_state: int, n_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_state, n_channels * 2),
            nn.GELU(),
            nn.Linear(n_channels * 2, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, residual_patches: torch.Tensor, state_vec: torch.Tensor) -> torch.Tensor:
        gate = self.gate(state_vec)
        return residual_patches * gate[:, None, None, :]


class SharedConvTrunk(nn.Module):
    def __init__(self, n_channels: int, d_trunk: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, d_trunk, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_trunk),
            nn.GELU(),
            nn.Conv1d(d_trunk, d_trunk, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_trunk),
            nn.GELU(),
            nn.Conv1d(d_trunk, d_trunk, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_trunk),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(patches.permute(0, 2, 1))
        return self.pool(hidden).squeeze(-1)


class ScaleHead(nn.Module):
    def __init__(self, d_trunk: int, d_z: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_trunk, d_z),
            nn.GELU(),
            nn.Linear(d_z, d_z),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)


class MultiScalePatchEncoder(nn.Module):
    def __init__(self, config: CoReMADConfig):
        super().__init__()
        self.config = config
        self.patch_sizes = config.patch_sizes
        self.channel_mod = ChannelModulation(config.d_state, config.n_channels)
        self.shared_trunk = SharedConvTrunk(config.n_channels, config.d_trunk)
        self.scale_heads = nn.ModuleList([ScaleHead(config.d_trunk, config.d_z) for _ in self.patch_sizes])

    @staticmethod
    def make_patches(seq: torch.Tensor, patch_size: int) -> torch.Tensor:
        batch, length, channels = seq.shape
        n_patches = length // patch_size
        trimmed = seq[:, : n_patches * patch_size, :]
        return trimmed.reshape(batch, n_patches, patch_size, channels)

    def forward(self, residual: torch.Tensor, raw_x: torch.Tensor, state_vec: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        batch = residual.shape[0]
        outputs = {
            "z": [],
            "u": [],
            "c": [],
            "r_patches": [],
            "r_mod_patches": [],
            "x_patches": [],
            "n_patches": [],
        }
        for scale_idx, patch_size in enumerate(self.patch_sizes):
            r_patch = self.make_patches(residual, patch_size)
            x_patch = self.make_patches(raw_x, patch_size)
            n_patches = r_patch.shape[1]

            if self.config.use_channel_modulation:
                r_mod = self.channel_mod(r_patch, state_vec)
            else:
                r_mod = r_patch
            r_feat = self.shared_trunk(r_mod.reshape(batch * n_patches, patch_size, self.config.n_channels))
            u = self.scale_heads[scale_idx](r_feat).reshape(batch, n_patches, self.config.d_z)
            c = self._build_context_keys(u)

            outputs["z"].append(u)
            outputs["u"].append(u)
            outputs["c"].append(c)
            outputs["r_patches"].append(r_patch)
            outputs["r_mod_patches"].append(r_mod)
            outputs["x_patches"].append(x_patch)
            outputs["n_patches"].append(n_patches)
        return outputs

    def _build_context_keys(self, u: torch.Tensor) -> torch.Tensor:
        radius = self.config.context_k
        ones = torch.ones(u.size(0), u.size(1), 1, device=u.device, dtype=u.dtype)
        u_pad = F.pad(u, (0, 0, radius, radius), mode="constant", value=0.0)
        o_pad = F.pad(ones, (0, 0, radius, radius), mode="constant", value=0.0)
        u_unfold = u_pad.unfold(1, 2 * radius + 1, 1)
        o_unfold = o_pad.unfold(1, 2 * radius + 1, 1)
        return u_unfold.sum(dim=-1) / o_unfold.sum(dim=-1).clamp(min=1.0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MaskedCompletionHead(nn.Module):
    def __init__(self, config: CoReMADConfig, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = patch_size * config.n_channels
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_z) * 0.02)
        self.pos_enc = PositionalEncoding(config.d_z)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_z,
            nhead=config.completion_n_heads,
            dim_feedforward=config.d_z * 2,
            dropout=config.completion_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.completion_n_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(config.d_z, config.d_z),
            nn.GELU(),
            nn.Linear(config.d_z, self.output_dim),
        )

    def forward(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = u.clone()
        expanded_mask = mask.unsqueeze(-1).expand_as(masked)
        token = self.mask_token.expand(u.size(0), u.size(1), -1)
        masked[expanded_mask] = token[expanded_mask]
        hidden = self.transformer(self.pos_enc(masked))
        return self.out_proj(hidden)


class NextPatchPredHead(nn.Module):
    def __init__(self, config: CoReMADConfig, patch_size: int):
        super().__init__()
        output_dim = patch_size * config.n_channels
        self.pred = nn.Sequential(
            nn.Linear(config.d_z, config.d_z),
            nn.GELU(),
            nn.Linear(config.d_z, output_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.pred(u[:, :-1, :])


class CoReMADModel(nn.Module):
    def __init__(self, config: CoReMADConfig):
        super().__init__()
        self.config = config
        self.decomposer = STSDDecomposer(
            n_channels=config.n_channels,
            seq_len=config.seq_len,
            d_hidden=config.stsd_hidden,
            lowpass_center=config.stsd_lowpass_center,
        )
        self.state_encoder = StateEncoder(
            n_channels=config.n_channels,
            d_state=config.d_state,
            d_hidden=config.state_hidden,
        )
        self.patch_encoder = MultiScalePatchEncoder(config)
        self.completion_heads = (
            nn.ModuleList([MaskedCompletionHead(config, patch_size) for patch_size in config.patch_sizes])
            if config.use_completion_head
            else nn.ModuleList()
        )
        self.prediction_heads = nn.ModuleList(
            [NextPatchPredHead(config, patch_size) for patch_size in config.patch_sizes]
        )

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if self.config.use_stsd_decomposition:
            slow, residual = self.decomposer(x)
            state_source = slow
        else:
            slow = x
            residual = x
            state_source = x
        state_vec = self.state_encoder(state_source)
        patch_outputs = self.patch_encoder(residual, x, state_vec)
        patch_outputs["slow"] = slow
        patch_outputs["residual"] = residual
        patch_outputs["state_vec"] = state_vec
        return patch_outputs

    def compute_pretraining_losses(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(x)
        total_mask_loss = x.new_tensor(0.0)
        total_pred_loss = x.new_tensor(0.0)
        n_scales = max(1, len(self.config.patch_sizes))

        for scale_idx, patch_size in enumerate(self.config.patch_sizes):
            u = encoded["u"][scale_idx]
            target = encoded["x_patches"][scale_idx]
            if self.config.use_completion_head:
                mask = self.sample_random_mask(u.size(0), u.size(1), u.device)
                comp_pred = self.completion_heads[scale_idx](u, mask).reshape(
                    u.size(0), u.size(1), patch_size, self.config.n_channels
                )
                comp_err = self._completion_error(comp_pred, target)
                total_mask_loss = total_mask_loss + (comp_err * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

            if u.size(1) > 1:
                pred_target = target[:, 1:, :, :].reshape(
                    u.size(0), u.size(1) - 1, patch_size * self.config.n_channels
                )
                pred_hat = self.prediction_heads[scale_idx](u)
                total_pred_loss = total_pred_loss + (pred_hat - pred_target).abs().mean()

        total_mask_loss = total_mask_loss / n_scales
        total_pred_loss = total_pred_loss / n_scales
        if self.config.use_stsd_decomposition:
            smooth_loss = self.decomposer.smoothness_loss(encoded["slow"])
        else:
            smooth_loss = x.new_tensor(0.0)
        total_loss = total_mask_loss + self.config.lambda_pred * total_pred_loss + self.config.lambda_smooth * smooth_loss
        return {
            "loss": total_loss,
            "mask_loss": total_mask_loss,
            "pred_loss": total_pred_loss,
            "smooth_loss": smooth_loss,
        }

    def deterministic_completion_scores(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], list[torch.Tensor]]:
        encoded = self.encode(x)
        if not self.config.use_completion_head:
            return encoded, []
        scores: list[torch.Tensor] = []
        for scale_idx, patch_size in enumerate(self.config.patch_sizes):
            u = encoded["u"][scale_idx]
            target = encoded["x_patches"][scale_idx]
            accum = x.new_zeros(u.size(0), u.size(1))
            counts = x.new_zeros(u.size(0), u.size(1))
            positions = torch.arange(u.size(1), device=x.device)
            for group in range(self.config.n_mask_groups):
                mask = (positions % self.config.n_mask_groups == group).unsqueeze(0).expand(u.size(0), -1)
                if not mask.any():
                    continue
                pred = self.completion_heads[scale_idx](u, mask).reshape(
                    u.size(0), u.size(1), patch_size, self.config.n_channels
                )
                err = self._completion_error(pred, target)
                accum = accum + err * mask.float()
                counts = counts + mask.float()
            scores.append(accum / counts.clamp(min=1.0))
        return encoded, scores

    @staticmethod
    def _completion_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_channel_err = (pred - target).abs().mean(dim=2)
        return per_channel_err.mean(dim=-1)

    def sample_random_mask(self, batch: int, n_patches: int, device: torch.device) -> torch.Tensor:
        n_mask = max(1, int(n_patches * self.config.mask_ratio))
        mask = torch.zeros(batch, n_patches, device=device, dtype=torch.bool)
        random_scores = torch.rand(batch, n_patches, device=device)
        indices = torch.topk(random_scores, k=n_mask, dim=1, largest=False).indices
        mask.scatter_(1, indices, True)
        return mask
