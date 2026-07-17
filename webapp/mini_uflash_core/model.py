from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DFlashOCRConfig:
    """Configuration for the lightweight mask-block drafter.

    ``max_block_size`` includes one clean anchor slot. Therefore B8 means
    one anchor representation plus seven masked draft positions.
    """

    target_hidden_size: int = 1280
    num_target_features: int = 5
    draft_dim: int = 320
    num_layers: int = 3
    num_heads: int = 5
    ff_dim: int = 768
    max_block_size: int = 8
    dropout: float = 0.0
    context_tokens: str = "fused_plus_layers"
    # DSpark-style low-rank Markov refine in hidden space (8GB-friendly; not V×V).
    markov_rank: int = 64
    use_markov: bool = True

    @property
    def max_draft_tokens(self) -> int:
        return self.max_block_size - 1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "DFlashOCRConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in value.items() if k in known}
        return cls(**filtered)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(scale + self.eps).to(dtype=x.dtype)
        return x * scale * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"draft_dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if active_mask is not None:
            if active_mask.shape != (batch, seq_len):
                raise ValueError(
                    f"active_mask must be {(batch, seq_len)}, got {tuple(active_mask.shape)}"
                )
            # SDPA bool mask: True means the query-key pair is allowed.
            allowed = active_mask[:, None, :, None] & active_mask[:, None, None, :]
            attn_mask = allowed

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        y = self.out(y)
        if active_mask is not None:
            y = y * active_mask.unsqueeze(-1).to(dtype=y.dtype)
        return y


class ContextKVAttention(nn.Module):
    """Cross attention whose target-derived K/V can be projected and cached."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"draft_dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def project_kv(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, mem_len, _ = context.shape
        k = self.k_proj(context).view(batch, mem_len, self.heads, self.head_dim)
        v = self.v_proj(context).view(batch, mem_len, self.heads, self.head_dim)
        return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim)
        q = q.transpose(1, 2)
        k, v = kv
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        y = self.out_proj(y)
        if active_mask is not None:
            y = y * active_mask.unsqueeze(-1).to(dtype=y.dtype)
        return y


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, ff_dim, bias=False)
        self.up = nn.Linear(dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))


class KVInjectedDraftLayer(nn.Module):
    def __init__(self, config: DFlashOCRConfig) -> None:
        super().__init__()
        d = config.draft_dim
        self.self_norm = RMSNorm(d)
        self.self_attn = MultiHeadSelfAttention(d, config.num_heads, config.dropout)
        self.context_norm = RMSNorm(d)
        self.context_attn = ContextKVAttention(d, config.num_heads, config.dropout)
        self.ff_norm = RMSNorm(d)
        self.ff = FeedForward(d, config.ff_dim, config.dropout)

    def project_context_kv(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.context_attn.project_kv(context)

    def forward(
        self,
        x: torch.Tensor,
        context_kv: Tuple[torch.Tensor, torch.Tensor],
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.self_norm(x), active_mask)
        x = x + self.context_attn(self.context_norm(x), context_kv, active_mask)
        x = x + self.ff(self.ff_norm(x)) * active_mask.unsqueeze(-1).to(x.dtype)
        return x


class MarkovHiddenHead(nn.Module):
    """DSpark-inspired first-order transition in *hidden* space.

    Avoids a V×V / V×r Markov matrix (Unlimited-OCR vocab ≈ 129k would be
    tens of millions of parameters). Instead, a rank-r residual is applied to
    the drafted hidden state conditioned on the previous token embedding.
    """

    def __init__(self, hidden_size: int, rank: int = 64) -> None:
        super().__init__()
        self.rank = int(rank)
        self.curr_down = nn.Linear(hidden_size, self.rank, bias=False)
        self.prev_down = nn.Linear(hidden_size, self.rank, bias=False)
        self.up = nn.Linear(self.rank * 2, hidden_size, bias=False)
        # Near-zero init so old pure-parallel checkpoints stay intact at load.
        nn.init.zeros_(self.up.weight)

    def bias(self, curr_hidden: torch.Tensor, prev_embedding: torch.Tensor) -> torch.Tensor:
        """Return residual to add to ``curr_hidden`` (same shape)."""
        z = torch.cat(
            [self.curr_down(curr_hidden), self.prev_down(prev_embedding)], dim=-1
        )
        return self.up(F.silu(z))


class MaskBlockDrafter(nn.Module):
    """DFlash-style OCR drafter (+ optional DSpark Markov refine).

    Input:
        target_features: [batch, 5, target_hidden_size] from the target state
            that predicts the clean anchor token.
        anchor_embedding: frozen target embedding of that clean anchor token.
        block_size: total block slots, including the anchor (4/6/8).

    Output:
        hidden: [batch, block_size - 1, target_hidden_size], consumed by the
            frozen target LM head to obtain token logits.
        acceptance_logits: one correctness logit per drafted position.

    All active positions attend bidirectionally within the block. Every layer
    receives target context through its own projected K/V entries. ``prepare_context``
    exposes those K/V tensors so an online integration can cache and reuse them.
    """

    def __init__(self, config: DFlashOCRConfig) -> None:
        super().__init__()
        self.config = config
        h = config.target_hidden_size
        d = config.draft_dim
        n = config.num_target_features

        self.feature_norm = RMSNorm(h)
        self.layer_feature_proj = nn.Linear(h, d, bias=False)
        self.fused_context_proj = nn.Linear(n * h, d, bias=False)
        self.anchor_proj = nn.Linear(h, d, bias=False)

        self.anchor_type = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.position = nn.Parameter(torch.randn(1, config.max_block_size, d) * 0.02)

        self.layers = nn.ModuleList(
            [KVInjectedDraftLayer(config) for _ in range(config.num_layers)]
        )
        self.out_norm = RMSNorm(d)
        self.hidden_head = nn.Linear(d, h, bias=False)
        # Cheap pre-draft gate: predicts per-position correctness directly from
        # fused target context, so the runtime can choose B1/B4/B6/B8 before
        # paying the full mask-block drafting cost.
        self.context_acceptance_head = nn.Linear(d, config.max_draft_tokens)
        # Post-draft diagnostic head; useful for calibration analysis.
        self.post_acceptance_head = nn.Linear(d, 1)
        # DSpark-style serial refine (optional; zero-init up-projection).
        self.markov_head = MarkovHiddenHead(h, rank=int(config.markov_rank))

        nn.init.normal_(self.fused_context_proj.weight, std=0.02 / (n ** 0.5))

    def _validate_inputs(self, target_features: torch.Tensor, block_size: int) -> None:
        if target_features.ndim != 3:
            raise ValueError(
                "target_features must be [batch, num_target_features, hidden_size]"
            )
        expected = (self.config.num_target_features, self.config.target_hidden_size)
        if tuple(target_features.shape[1:]) != expected:
            raise ValueError(
                f"Expected feature shape (*, {expected[0]}, {expected[1]}), "
                f"got {tuple(target_features.shape)}"
            )
        if not 2 <= block_size <= self.config.max_block_size:
            raise ValueError(
                f"block_size must be in [2, {self.config.max_block_size}], got {block_size}"
            )

    def fuse_context(self, target_features: torch.Tensor) -> torch.Tensor:
        normed = self.feature_norm(target_features)
        per_layer = self.layer_feature_proj(normed)
        fused = self.fused_context_proj(normed.flatten(start_dim=1)).unsqueeze(1)
        if self.config.context_tokens == "fused_only":
            return fused
        if self.config.context_tokens != "fused_plus_layers":
            raise ValueError(f"Unknown context_tokens={self.config.context_tokens!r}")
        return torch.cat([fused, per_layer], dim=1)

    def prepare_context(
        self, target_features: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Fuse target features and project per-layer K/V tensors once."""
        context = self.fuse_context(target_features)
        cache = [layer.project_context_kv(context) for layer in self.layers]
        return context, cache

    def predict_acceptance_from_context(self, context: torch.Tensor) -> torch.Tensor:
        """Return correctness logits for all maximum draft positions."""
        if context.ndim != 3 or context.shape[-1] != self.config.draft_dim:
            raise ValueError("context must be [batch, memory_tokens, draft_dim]")
        return self.context_acceptance_head(context[:, 0, :])

    def predict_acceptance(self, target_features: torch.Tensor) -> torch.Tensor:
        """Cheap gate path used before selecting the draft block size."""
        context = self.fuse_context(target_features)
        return self.predict_acceptance_from_context(context)

    def _build_block_input(
        self,
        target_features: torch.Tensor,
        block_size: int,
        anchor_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = target_features.shape[0]
        max_slots = self.config.max_block_size
        anchor_hidden = target_features[:, -1, :] if anchor_embedding is None else anchor_embedding
        if anchor_hidden.shape != (batch, self.config.target_hidden_size):
            raise ValueError(
                f"anchor_embedding must be {(batch, self.config.target_hidden_size)}, "
                f"got {tuple(anchor_hidden.shape)}"
            )
        anchor = self.anchor_proj(self.feature_norm(anchor_hidden)).unsqueeze(1)
        anchor = anchor + self.anchor_type
        masks = self.mask_token.expand(batch, max_slots - 1, -1)
        x = torch.cat([anchor, masks], dim=1) + self.position

        active_mask = torch.arange(max_slots, device=x.device)[None, :] < block_size
        active_mask = active_mask.expand(batch, -1)
        x = x * active_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x, active_mask

    def forward(
        self,
        target_features: torch.Tensor,
        block_size: Optional[int] = None,
        context_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        anchor_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        block_size = block_size or self.config.max_block_size
        self._validate_inputs(target_features, block_size)
        x, active_mask = self._build_block_input(
            target_features, block_size, anchor_embedding=anchor_embedding
        )

        context = self.fuse_context(target_features)
        if context_kv_cache is None:
            context_kv_cache = [
                layer.project_context_kv(context) for layer in self.layers
            ]
        if len(context_kv_cache) != len(self.layers):
            raise ValueError("context_kv_cache length does not match num_layers")

        for layer, kv in zip(self.layers, context_kv_cache):
            x = layer(x, kv, active_mask)

        draft_len = block_size - 1
        draft_states = self.out_norm(x[:, 1:block_size, :])
        hidden = self.hidden_head(draft_states)
        acceptance_logits = self.predict_acceptance_from_context(context)[:, :draft_len]
        post_acceptance_logits = self.post_acceptance_head(draft_states).squeeze(-1)
        return {
            "hidden": hidden,
            "acceptance_logits": acceptance_logits,
            "post_acceptance_logits": post_acceptance_logits,
            "draft_states": draft_states,
            "active_mask": active_mask[:, 1:block_size],
        }

    def refine_with_markov(
        self,
        hidden: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_embeddings: Optional[torch.Tensor] = None,
        input_embeddings: Optional[nn.Module] = None,
        anchor_embedding: Optional[torch.Tensor] = None,
        teacher_force: bool = True,
    ) -> torch.Tensor:
        """Apply left-to-right Markov residual on drafted hiddens.

        Training (teacher_force=True): condition on gold previous tokens.
        Inference: condition on previously *sampled* draft tokens; pass
        ``prev_token_ids`` as the already-chosen prefix (length grows).
        """
        if not bool(getattr(self.config, "use_markov", True)):
            return hidden
        if not hasattr(self, "markov_head"):
            return hidden
        batch, draft_len, _ = hidden.shape
        refined: List[torch.Tensor] = []
        for k in range(draft_len):
            curr = hidden[:, k, :]
            if k == 0:
                # First draft position: optional light bias from the clean anchor.
                if anchor_embedding is not None:
                    curr = curr + self.markov_head.bias(curr, anchor_embedding)
                refined.append(curr)
                continue
            if prev_embeddings is not None:
                prev_emb = prev_embeddings[:, k - 1, :]
            elif prev_token_ids is not None and input_embeddings is not None:
                prev_emb = input_embeddings(prev_token_ids[:, k - 1])
            elif teacher_force and prev_token_ids is not None and input_embeddings is not None:
                prev_emb = input_embeddings(prev_token_ids[:, k - 1])
            else:
                refined.append(curr)
                continue
            refined.append(curr + self.markov_head.bias(curr, prev_emb))
        return torch.stack(refined, dim=1)

    @torch.inference_mode()
    def draft(
        self,
        target_features: torch.Tensor,
        lm_head: nn.Module,
        block_size: Optional[int] = None,
        anchor_embedding: Optional[torch.Tensor] = None,
        input_embeddings: Optional[nn.Module] = None,
        use_markov: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self.forward(
            target_features, block_size=block_size, anchor_embedding=anchor_embedding
        )
        hidden = output["hidden"]
        draft_len = hidden.shape[1]
        apply_markov = (
            bool(self.config.use_markov) if use_markov is None else bool(use_markov)
        )
        apply_markov = apply_markov and hasattr(self, "markov_head") and input_embeddings is not None

        if not apply_markov:
            logits = lm_head(hidden)
            tokens = logits.argmax(dim=-1)
        else:
            # Sequential sample with Markov residual (DSpark semi-AR).
            token_list: List[torch.Tensor] = []
            logit_list: List[torch.Tensor] = []
            refined_h: List[torch.Tensor] = []
            for k in range(draft_len):
                curr = hidden[:, k, :]
                if k == 0:
                    if anchor_embedding is not None:
                        curr = curr + self.markov_head.bias(curr, anchor_embedding)
                else:
                    prev_emb = input_embeddings(token_list[-1])
                    curr = curr + self.markov_head.bias(curr, prev_emb)
                logit_k = lm_head(curr.unsqueeze(1)).squeeze(1)
                tok_k = logit_k.argmax(dim=-1)
                logit_list.append(logit_k)
                token_list.append(tok_k)
                refined_h.append(curr)
            logits = torch.stack(logit_list, dim=1)
            tokens = torch.stack(token_list, dim=1)
            hidden = torch.stack(refined_h, dim=1)

        gate_p = torch.sigmoid(output["acceptance_logits"].float())
        post_p = torch.sigmoid(output["post_acceptance_logits"].float())
        # Blend pre/post heads for scheduling; clamp for numerical safety.
        correctness_probability = (0.55 * post_p + 0.45 * gate_p).clamp(1e-4, 1.0 - 1e-4)
        survival = correctness_probability.cumprod(dim=-1)
        expected_accepted = survival.sum(dim=-1)
        return {
            **output,
            "hidden": hidden,
            "logits": logits,
            "tokens": tokens,
            "correctness_probability": correctness_probability,
            "post_probability": post_p,
            "gate_probability": gate_p,
            "survival": survival,
            "expected_accepted": expected_accepted,
        }

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
