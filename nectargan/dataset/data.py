from dataclasses import dataclass
from typing import Any

@dataclass
class TextEmbeddedMetadata:
    schema_version: int
    total_captions: int
    total_images:   int
    items: dict[str, dict[str, Any]]

