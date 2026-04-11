"""Datasets for image-caption training pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class DatasetIssue:
    row: int
    file_name: str
    reason: str


class ImageCaptionDataset(Dataset):
    """Loads image-caption pairs from `images/` + `metadata.jsonl` layout.

    metadata.jsonl rows should include:
    - file_name: relative filename under images/
    - text: image caption

    A `webdataset` mode is reserved for future support and currently raises
    a clear NotImplementedError.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        strict: bool = False,
        allow_empty_caption: bool = False,
        source_type: str = "folder",
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.strict = strict
        self.allow_empty_caption = allow_empty_caption
        self.source_type = source_type

        if self.source_type == "webdataset":
            raise NotImplementedError("WebDataset shard loading is planned but not implemented yet.")

        self.images_dir = self.root / "images"
        self.metadata_path = self.root / "metadata.jsonl"
        self.issues: List[DatasetIssue] = []
        self.samples: List[Dict[str, str]] = []
        self._load_folder_samples()

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in {self.root}. "
                "Expected `images/` and `metadata.jsonl` with `file_name` and `text` fields."
            )

    def _load_folder_samples(self) -> None:
        if not self.images_dir.exists() or not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self._add_issue(i, "", "invalid_json")
                    continue

                file_name = str(record.get("file_name", "")).strip()
                caption = str(record.get("text", "")).strip()

                if not file_name:
                    self._add_issue(i, file_name, "missing_file_name")
                    continue
                if (not caption) and (not self.allow_empty_caption):
                    self._add_issue(i, file_name, "missing_caption")
                    continue

                image_path = (self.images_dir / file_name).resolve()
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    self._add_issue(i, file_name, "unsupported_extension")
                    continue
                if not image_path.exists():
                    self._add_issue(i, file_name, "missing_image_file")
                    continue

                self.samples.append({"image_path": str(image_path), "text": caption})

        if self.strict and self.issues:
            problems = "\n".join(
                f"line={issue.row} file={issue.file_name!r} reason={issue.reason}" for issue in self.issues[:20]
            )
            raise ValueError(f"Dataset integrity checks failed:\n{problems}")

    def _add_issue(self, row: int, file_name: str, reason: str) -> None:
        self.issues.append(DatasetIssue(row=row, file_name=file_name, reason=reason))

    def validate_integrity(self) -> List[DatasetIssue]:
        """Return list of dataset issues found during load."""
        return self.issues

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "text": sample["text"],
            "image_path": sample["image_path"],
        }
