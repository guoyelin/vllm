# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model-family request metadata the edge extracts at admission.

Only the edge holds the real prompt. Model families whose cloud-side forward
reads more than the prompt embeddings get that extra input derived here, once
per request, in a form that carries no token id: M-RoPE positions for the
Qwen-VL family; the Engram hash parameters for DeepSeek V4.1. The edge
scheduler and engine use the family selected for the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_model.lwd_engram_hasher import LwdEngramHasher

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.request import Request

    LwdMropePositionsFn = Callable[..., tuple[torch.Tensor, Any]]

_LWD_DEEPSEEK_V41_ARCHITECTURE = "DeepseekV41ForCausalLM"


class LwdModelFamily:
    """Model family whose cloud reads the prompt embeddings only: nothing to
    extract. Subclasses override what their cloud forward needs."""

    def admit(self, request: Request) -> None:
        """Precompute at admission what the range notifies of `request`
        slice."""

    def has_mrope_positions(self, request_id: str) -> bool:
        """Whether the request carries M-RoPE positions. Read per range
        item, so it must not serialize them."""
        return False

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        """Raw int64 `[3, num_prompt_tokens]` M-RoPE positions of the whole
        prompt; None when the request carries none."""
        return None

    def release(self, request_id: str) -> None:
        """Drop what `admit` kept for the request."""

    def held_request_ids(self) -> list[str]:
        """Requests that still hold admission state."""
        return []

    def engram_fingerprint(self) -> str:
        """Digest of the Engram hash parameters; empty without Engram."""
        return ""


class LwdMropeModelFamily(LwdModelFamily):
    """Qwen-VL family: the cloud's M-RoPE positions of multimodal prompts.

    Positions need the real prompt and the image grids, so the edge computes
    them for the whole prompt at admission and the request notify carries
    them whole.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self._vllm_config = vllm_config
        # Resolved on the first multimodal request; model classes lacking it
        # fail fast there.
        self._positions_fn: LwdMropePositionsFn | None = None
        # Whole-prompt [3, N] CPU positions per multimodal request.
        self._positions: dict[str, torch.Tensor] = {}

    def admit(self, request: Request) -> None:
        if not _lwd_request_has_mm(request):
            return
        assert request.prompt_token_ids is not None
        mm_features = [f for f in request.mm_features if f.modality != "prompt_embeds"]
        positions, _delta = self._resolve_positions_fn(request)(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mm_features,
            config=self._vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._positions[request.request_id] = positions

    def _resolve_positions_fn(self, request: Request) -> LwdMropePositionsFn:
        if self._positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_config = self._vllm_config.model_config
            model_cls, _ = ModelRegistry.resolve_model_cls(
                model_config.architectures, model_config
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions (Qwen3VL family, incl. "
                    f"Qwen3.5); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._positions_fn = fn
        return self._positions_fn

    def has_mrope_positions(self, request_id: str) -> bool:
        return request_id in self._positions

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        positions = self._positions.get(request_id)
        if positions is None:
            return None
        return positions.contiguous().numpy().tobytes()

    def release(self, request_id: str) -> None:
        self._positions.pop(request_id, None)

    def held_request_ids(self) -> list[str]:
        return list(self._positions)


class LwdDeepseekV41ModelFamily(LwdModelFamily):
    """DeepSeek V4.1: the Engram hash parameters of the deployment.

    The edge hashes the prompt into the bucket indices the cloud's Engram
    tables are addressed by, so both sides must derive the hash parameters
    from the same model and tokenizer. The family holds the hasher for the
    handshake digest alone; the hashing itself runs in the edge worker,
    where the prompt rows are.
    """

    def __init__(self, engram_hasher: LwdEngramHasher | None) -> None:
        self._engram_hasher = engram_hasher

    def engram_fingerprint(self) -> str:
        if self._engram_hasher is None:
            return ""
        return self._engram_hasher.fingerprint()


def _lwd_request_has_mm(request: Request) -> bool:
    """Whether the request carries real multimodal data (prompt_embeds
    passthrough does not count -- it has no grid metadata, mrope is
    handled as text positions, consistent with the native definition)."""
    return any(f.modality != "prompt_embeds" for f in request.mm_features)


def lwd_select_model_family(vllm_config: VllmConfig) -> LwdModelFamily:
    """The request metadata family of the served model."""
    model_config = vllm_config.model_config
    family: LwdModelFamily
    if model_config.architecture == _LWD_DEEPSEEK_V41_ARCHITECTURE:
        family = LwdDeepseekV41ModelFamily(
            LwdEngramHasher.from_model_config(model_config)
        )
    elif model_config.uses_mrope:
        family = LwdMropeModelFamily(vllm_config)
    else:
        family = LwdModelFamily()
    lwd_logger.info("[LWD] request metadata family: %s", type(family).__name__)
    return family


def lwd_cloud_engram_fingerprint(model_config: ModelConfig) -> str:
    """Digest of the cloud's Engram hash parameters, derived the way the edge
    family derives its own; empty without Engram."""
    if model_config.architecture != _LWD_DEEPSEEK_V41_ARCHITECTURE:
        return ""
    hasher = LwdEngramHasher.from_model_config(model_config)
    return hasher.fingerprint() if hasher is not None else ""
