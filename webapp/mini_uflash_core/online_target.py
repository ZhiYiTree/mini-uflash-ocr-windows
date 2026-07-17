"""Online target wrapper for the Mini UFlash exactness probe.

Ported from ``stage9_v2_b8_exactness_probe.py:OnlineTarget`` in the Mini
UFlash Stage 9 research code. This class wraps the Unlimited-OCR target model
and provides:

* ``prefill`` — re-initialise the KV cache from the prompt + image, returning
  features, logits, and cache state.
* ``step`` — a single q_len=1 decode step (the authoritative strict-B1
  trajectory).
* ``verify_block`` — a q_len=Block decode for diagnostic block verification.

All methods are ``@torch.inference_mode()`` and use the model's own autocast
dtype. ``synchronize`` is called before/after GPU work for accurate latency
measurement on Windows.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Sequence

import torch
from transformers.cache_utils import DynamicCache  # type: ignore

from .stage9_common import (
    FeatureTap,
    extend_image_mask,
    synchronize,
)


class OnlineTarget:
    """Wraps the target LM for q_len=1 / q_len=Block stepping."""

    def __init__(
        self,
        model: torch.nn.Module,
        tap: FeatureTap,
        tensors: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        self.model = model
        self.tap = tap
        self.tensors = tensors
        self.device = device
        parameter_dtype = next(model.parameters()).dtype
        self.autocast_dtype = (
            parameter_dtype
            if parameter_dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
        )
        self.autocast_enabled = (
            device.type == "cuda"
            and parameter_dtype in (torch.float16, torch.bfloat16)
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        )

    @torch.inference_mode()
    def prefill(self, generated: Sequence[int]) -> Dict[str, Any]:
        """Full prefill with prompt + image + optional suffix tokens.

        Returns dict with cache, next_logits, features, absolute_length,
        physical_cache_length, seconds.
        """
        suffix = torch.tensor([list(generated)], dtype=torch.long, device=self.device)
        sequence = torch.cat([self.tensors["prompt_ids"], suffix], dim=1)
        image_mask = extend_image_mask(
            self.tensors["base_image_mask"], int(sequence.shape[1])
        ).to(self.device)
        self.tap.clear()
        synchronize(self.device)
        started = time.perf_counter()
        initial_cache = DynamicCache()
        with self._autocast():
            output = self.model(
                input_ids=sequence,
                attention_mask=torch.ones_like(sequence, dtype=torch.long),
                images=[
                    (self.tensors["images_crop"], self.tensors["images_ori"])
                ],
                images_seq_mask=image_mask,
                images_spatial_crop=self.tensors["spatial_crop"],
                past_key_values=initial_cache,
                use_cache=True,
                return_dict=True,
            )
        synchronize(self.device)
        return {
            "cache": output.past_key_values,
            "logits": output.logits,
            "next_logits": output.logits[:, -1, :],
            "features": self.tap.last_features(),
            "absolute_length": int(sequence.shape[1]),
            "physical_cache_length": _cache_seq_length(output.past_key_values),
            "seconds": time.perf_counter() - started,
        }

    @torch.inference_mode()
    def step(
        self, token_id: int, cache: Any, absolute_length: int
    ) -> Dict[str, Any]:
        """Single q_len=1 decode step (the strict baseline trajectory)."""
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        self.tap.clear()
        synchronize(self.device)
        started = time.perf_counter()
        with self._autocast():
            output = self.model(
                input_ids=token,
                attention_mask=None,
                position_ids=torch.tensor(
                    [[absolute_length]], dtype=torch.long, device=self.device
                ),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        synchronize(self.device)
        return {
            "cache": output.past_key_values,
            "next_logits": output.logits[:, -1, :],
            "features": self.tap.last_features(),
            "absolute_length": absolute_length + 1,
            "physical_cache_length": _cache_seq_length(output.past_key_values),
            "seconds": time.perf_counter() - started,
        }

    @torch.inference_mode()
    def verify_block(
        self,
        anchor_id: int,
        draft_ids: Sequence[int],
        cache: Any,
        absolute_length: int,
        full_prefix: Sequence[int],
        policy: Dict[str, int],
    ) -> Dict[str, Any]:
        """Run q_len=Block verification and return a committable cache state.

        Exactness-probe callers may use only ``predictions``. The explicitly
        authorised Direct Block experiment also consumes the returned cache,
        logits and features when an entire draft block is accepted.
        """
        block = torch.tensor(
            [[int(anchor_id), *[int(x) for x in draft_ids]]],
            dtype=torch.long,
            device=self.device,
        )
        block_len = int(block.shape[1])
        physical_before = _cache_seq_length(cache)
        self.tap.clear()
        synchronize(self.device)
        started = time.perf_counter()
        with self._autocast():
            output = self.model(
                input_ids=block,
                attention_mask=None,
                position_ids=torch.arange(
                    absolute_length,
                    absolute_length + block_len,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        synchronize(self.device)
        block_logits = output.logits[:, : len(draft_ids), :]
        predictions = []
        for position in range(len(draft_ids)):
            processor_prefix = [
                *[int(x) for x in full_prefix],
                int(anchor_id),
                *[int(x) for x in draft_ids[:position]],
            ]
            predictions.append(
                policy_argmax(block_logits[:, position, :], processor_prefix, policy)
            )
        return {
            "predictions": predictions,
            "cache": output.past_key_values,
            "logits": output.logits,
            "next_logits": output.logits[:, -1, :],
            "features": self.tap.last_features(),
            "absolute_length": absolute_length + block_len,
            "seconds": time.perf_counter() - started,
            "physical_cache_length_before": physical_before,
            "physical_cache_length_after": _cache_seq_length(output.past_key_values),
        }


# Convenience re-exports so the engine can do:
#   from webapp.mini_uflash_core import ...
# instead of drilling into stage9_common.
from .stage9_common import (  # noqa: E402, F401
    clone_cache,
    cache_seq_length,
    cache_tensors_are_disjoint,
    first_mismatch,
    get_input_embeddings,
    get_output_embeddings,
    payload_tensors,
    policy_argmax,
    generation_policy,
    atomic_json,
)

# Internal shortcut used inside this module.
_cache_seq_length = cache_seq_length
