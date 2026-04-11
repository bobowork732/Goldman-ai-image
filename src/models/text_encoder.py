"""Text encoder wrappers for latent diffusion conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import importlib

import torch
from torch import nn

_transformers = importlib.import_module("transformers") if importlib.util.find_spec("transformers") else None
AutoModel = getattr(_transformers, "AutoModel", None)
AutoTokenizer = getattr(_transformers, "AutoTokenizer", None)


class SimpleTokenizer:
    """Fallback whitespace tokenizer with dynamic vocabulary."""

    def __init__(self, max_length: int = 77) -> None:
        self.max_length = max_length
        self.vocab = {"<pad>": 0, "<unk>": 1}

    def encode(self, texts: List[str]) -> torch.Tensor:
        ids = []
        for text in texts:
            tokens = text.lower().split()
            row = []
            for tok in tokens[: self.max_length]:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
                row.append(self.vocab.get(tok, 1))
            row += [0] * (self.max_length - len(row))
            ids.append(row)
        return torch.tensor(ids, dtype=torch.long)


class TinyTextEncoder(nn.Module):
    """Minimal transformer-like text encoder when HF models are unavailable."""

    def __init__(self, vocab_size: int = 8192, embed_dim: int = 768, max_length: int = 77) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        h = self.token_embedding(input_ids) + self.pos_embedding(positions)
        return self.encoder(h)


@dataclass
class TextEncoderOutput:
    embeddings: torch.Tensor


class TextEncoder(nn.Module):
    """CLIP-like compatible text encoder with optional Hugging Face backend."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 77,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.hf_model = None
        self.hf_tokenizer = None
        self.fallback_tokenizer: Optional[SimpleTokenizer] = None

        if model_name and AutoTokenizer is not None and AutoModel is not None:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModel.from_pretrained(model_name)
            self.embed_dim = self.hf_model.config.hidden_size
        else:
            self.fallback_tokenizer = SimpleTokenizer(max_length=max_length)
            self.fallback_model = TinyTextEncoder(embed_dim=embed_dim, max_length=max_length)
            self.embed_dim = embed_dim

    def tokenize_to_ids(self, texts: List[str]) -> torch.Tensor:
        if self.hf_tokenizer is not None:
            batch = self.hf_tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return batch["input_ids"]
        assert self.fallback_tokenizer is not None
        return self.fallback_tokenizer.encode(texts)

    def tokenize(self, texts: List[str], device: torch.device) -> torch.Tensor:
        return self.tokenize_to_ids(texts).to(device)

    def encode_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.hf_model is not None:
            outputs = self.hf_model(input_ids=input_ids)
            return outputs.last_hidden_state
        return self.fallback_model(input_ids)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        input_ids = self.tokenize(texts, device)
        return self.encode_tokens(input_ids)

    def forward(self, texts: List[str], device: torch.device) -> TextEncoderOutput:
        return TextEncoderOutput(embeddings=self.encode_text(texts, device))
