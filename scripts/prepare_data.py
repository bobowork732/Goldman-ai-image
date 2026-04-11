"""Validate dataset integrity and preview tokenized training batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.collate import FixedLengthTokenizer, build_collate_fn
from src.data.dataset import ImageCaptionDataset
from src.data.transforms import build_image_transforms
from src.models.text_encoder import TextEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and preview diffusion training data")
    parser.add_argument("--data-root", type=str, required=True, help="Path containing images/ and metadata.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-preview-batches", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="Fail if any metadata issues are found")
    parser.add_argument("--source-type", type=str, default="folder", choices=["folder", "webdataset"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transforms = build_image_transforms(resolution=args.resolution, center_crop=False, random_flip=False)
    dataset = ImageCaptionDataset(
        root=args.data_root,
        transform=transforms,
        strict=args.strict,
        source_type=args.source_type,
    )

    issues = dataset.validate_integrity()
    print(f"Loaded {len(dataset)} valid samples from {Path(args.data_root).resolve()}")
    print(f"Integrity issues detected: {len(issues)}")
    for issue in issues[:10]:
        print(f"  - line={issue.row} file={issue.file_name!r} reason={issue.reason}")

    text_encoder = TextEncoder(model_name=None, max_length=args.max_length)
    tokenizer = FixedLengthTokenizer(text_encoder.tokenize_to_ids, max_length=args.max_length)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_collate_fn(tokenizer),
    )

    for i, batch in enumerate(loader):
        if i >= args.num_preview_batches:
            break
        print(f"\nPreview batch {i + 1}")
        print(f"  pixel_values: {tuple(batch['pixel_values'].shape)} {batch['pixel_values'].dtype}")
        print(f"  input_ids: {tuple(batch['input_ids'].shape)} {batch['input_ids'].dtype}")
        print(f"  attention_mask: {tuple(batch['attention_mask'].shape)}")
        print(f"  example text: {batch['text'][0]!r}")
        print(f"  example path: {batch['image_path'][0]}")

    # sanity check token lengths
    token_lengths = (batch["attention_mask"].sum(dim=1) for batch in loader)
    first_lengths = next(token_lengths)
    assert torch.all(first_lengths <= args.max_length), "Token lengths exceed configured max_length"


if __name__ == "__main__":
    main()
