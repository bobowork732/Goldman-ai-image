"""Collate helpers for diffusion training batches."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import torch


class FixedLengthTokenizer:
    """Tokenizer adapter returning fixed-length token IDs and attention masks."""

    def __init__(self, tokenizer: Callable[[List[str]], torch.Tensor], max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        ids = self.tokenizer(list(texts))
        if ids.shape[1] != self.max_length:
            if ids.shape[1] > self.max_length:
                ids = ids[:, : self.max_length]
            else:
                pad = torch.zeros(ids.shape[0], self.max_length - ids.shape[1], dtype=ids.dtype)
                ids = torch.cat([ids, pad], dim=1)

        attention_mask = (ids != 0).long()
        return {"input_ids": ids.long(), "attention_mask": attention_mask}


def build_collate_fn(tokenizer: FixedLengthTokenizer):
    """Build collate function combining images and fixed-length tokenized captions."""

    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
        texts = [str(item["text"]) for item in batch]
        paths = [str(item.get("image_path", "")) for item in batch]
        tokenized = tokenizer(texts)

        return {
            "pixel_values": pixel_values,
            "text": texts,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "image_path": paths,
        }

    return collate_fn
