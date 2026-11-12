"""MLX-native wav2vec2 CTC encoder used by the forced-alignment backend.

The encoder itself comes from ``mlx-audio`` (``mlx_audio.stt.models.wav2vec``);
this module only adds the CTC head, a loader that never touches pickle-based
``.bin`` checkpoints, and the vocabulary reader. Nothing here imports torch.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig
from mlx_audio.stt.models.wav2vec.wav2vec import Wav2Vec2Model

__all__ = [
    "MLXWav2Vec2CTC",
    "UnsupportedCheckpointError",
    "load_mlx_align_model",
    "load_vocab",
    "resolve_checkpoint",
]

WEIGHTS_FILENAME = "model.safetensors"

# config.json / vocab.json / preprocessor_config.json plus the safetensors
# weights; ``*.model`` and ``*.txt`` cover sentencepiece / vocab side files.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]


class UnsupportedCheckpointError(RuntimeError):
    """The checkpoint exists but cannot be run by the MLX backend.

    Raised for repos that ship no safetensors weights and for configs the MLX
    encoder does not implement, so ``--align_backend auto`` can fall back to
    the torch path.
    """


class MLXWav2Vec2CTC(nn.Module):
    """``Wav2Vec2ForCTC`` equivalent: wav2vec2 encoder plus a linear CTC head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def __call__(self, input_values: mx.array) -> mx.array:
        hidden_states = self.wav2vec2(input_values).last_hidden_state
        return self.lm_head(hidden_states)

    def sanitize(self, weights):
        """Rewrite a ``transformers`` ``Wav2Vec2ForCTC`` state dict for MLX.

        The encoder keys go through ``Wav2Vec2Model.sanitize`` (prefix strip,
        weight-norm rename, conv transpose). That helper drops ``lm_head.*``,
        so the CTC head is kept aside and re-attached afterwards.
        """
        head = {k: v for k, v in weights.items() if k.startswith("lm_head.")}
        encoder = {k: v for k, v in weights.items() if not k.startswith("lm_head.")}

        # ``Wav2Vec2Model.sanitize`` strips the ``wav2vec2.`` prefix because the
        # standalone encoder is the root module. Here the encoder lives under
        # ``self.wav2vec2``, so put the prefix back on.
        sanitized = {f"wav2vec2.{k}": v for k, v in self.wav2vec2.sanitize(encoder).items()}
        sanitized.update(head)
        return sanitized

    @classmethod
    def from_pretrained_local(cls, model_dir: Path) -> "MLXWav2Vec2CTC":
        """Build the model from a local snapshot directory (safetensors only)."""
        model_dir = Path(model_dir)

        with open(model_dir / "config.json", encoding="utf-8") as fh:
            config = ModelConfig.from_dict(json.load(fh))

        # The MLX encoder has no adapter layers; checkpoints that ship them
        # would fail the strict weight load with a much less useful message.
        if getattr(config, "adapter_attn_dim", None):
            raise UnsupportedCheckpointError(
                "checkpoint uses wav2vec2 attention adapters, which the MLX "
                "encoder does not implement"
            )

        model = cls(config)
        weights = mx.load(str(model_dir / WEIGHTS_FILENAME), format="safetensors")
        # fp32 is the parity baseline; some community checkpoints ship fp16.
        weights = {
            k: v.astype(mx.float32) if v.dtype != mx.float32 else v for k, v in weights.items()
        }
        try:
            model.load_weights(list(model.sanitize(weights).items()))
        except ValueError as e:
            raise UnsupportedCheckpointError(
                f"weights do not match the MLX wav2vec2 encoder ({e})"
            ) from e

        mx.eval(model.parameters())
        # MLX modules start in training mode, so every nn.Dropout would fire
        # (checkpoints such as wav2vec2-base-960h set *_dropout to 0.1).
        model.eval()
        return model


def resolve_checkpoint(repo_id: str, cache_dir=None, local_only: bool = False) -> Path:
    """Download (or locate) a HF snapshot and return its directory."""
    path = Path(
        snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            local_files_only=local_only,
            allow_patterns=ALLOW_PATTERNS,
        )
    )

    if not (path / WEIGHTS_FILENAME).is_file():
        raise UnsupportedCheckpointError(
            f"{repo_id} ships no {WEIGHTS_FILENAME} (pickle-only checkpoint); "
            "the MLX alignment backend never reads .bin weights — "
            "re-run with --align_backend torch"
        )

    return path


def load_vocab(model_dir) -> dict[str, int]:
    """Read ``vocab.json`` as a lower-cased ``token -> id`` dictionary."""
    vocab_path = Path(model_dir) / "vocab.json"
    if not vocab_path.is_file():
        raise UnsupportedCheckpointError(f"no vocab.json next to the weights in {model_dir}")

    with open(vocab_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # transformers writes a flat {token: id} mapping. Guard against a nested
    # container so a surprise layout degrades instead of exploding.
    if len(raw) == 1:
        only_value = next(iter(raw.values()))
        if isinstance(only_value, dict) and "id" not in only_value:
            raw = only_value

    vocab: dict[str, int] = {}
    for token, code in raw.items():
        if isinstance(code, dict):
            code = code.get("id")
        if code is None:
            continue
        vocab[str(token).lower()] = int(code)

    return vocab


def load_mlx_align_model(
    language_code: str,
    repo_id: str,
    model_dir=None,
    model_cache_only: bool = False,
):
    """Load a wav2vec2 CTC checkpoint on the MLX backend.

    Returns ``(model, metadata)`` with the same metadata shape as the torch
    paths, except ``metadata["type"] == "mlx"``.
    """
    path = resolve_checkpoint(repo_id, cache_dir=model_dir, local_only=model_cache_only)
    try:
        model = MLXWav2Vec2CTC.from_pretrained_local(path)
        vocab = load_vocab(path)
    except UnsupportedCheckpointError as e:
        raise UnsupportedCheckpointError(
            f"{repo_id} cannot be used by the MLX alignment backend ({e}); "
            "re-run with --align_backend torch"
        ) from e

    metadata = {
        "language": language_code,
        "dictionary": vocab,
        "type": "mlx",
    }
    return model, metadata
