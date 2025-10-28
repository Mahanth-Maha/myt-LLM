import os
import json
import logging
import torch
from pathlib import Path
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        shard_folder: str,
        context_length,
        shard_size=1_026_672_873,
        resume_info=None
    ):
        self.shard_folder = Path(shard_folder)
        self.context_length = context_length
        self.shard_size = shard_size
        self.resume_info = resume_info or {}

        self.shard_files = self._discover_shards()
        self.shard_index = self._build_shard_index()

        self.current_shard_idx = self.resume_info.get('shard_index', 0)
        self.current_offset = self.resume_info.get('offset', 0)

        logger.info(f"Found {len(self.shard_files)} shard files")
        logger.info(
            f"Total estimated tokens: {len(self.shard_files) * self.shard_size:,}")

        if self.resume_info:
            logger.info(
                f"Resuming from shard {self.current_shard_idx}, offset {self.current_offset}")

    def _discover_shards(self):
        shard_files = []

        if not self.shard_folder.exists():
            raise FileNotFoundError(
                f"Shard folder not found: {self.shard_folder}")

        for shard_file in self.shard_folder.rglob("shard_*.pt"):
            if shard_file.is_file():
                shard_files.append(shard_file)

        shard_files.sort(key=lambda x: (x.parent.name, x.name))

        if not shard_files:
            raise ValueError(f"No shard files found in {self.shard_folder}")

        return shard_files

    def _build_shard_index(self):
        index_file = self.shard_folder / "shard_index.json"

        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not load shard index: {e}. Rebuilding...")

        shard_index = {
            "total_files": len(self.shard_files),
            "estimated_total_tokens": len(self.shard_files) * self.shard_size,
            "files": {}
        }

        for i, shard_file in enumerate(self.shard_files):
            rel_path = str(shard_file.relative_to(self.shard_folder))
            shard_index["files"][rel_path] = {
                "index": i,
                "estimated_tokens": self.shard_size,
                "size_bytes": shard_file.stat().st_size if shard_file.exists() else 0
            }

        try:
            with open(index_file, 'w') as f:
                json.dump(shard_index, f, indent=2)
            logger.info(f"Saved shard index to {index_file}")
        except IOError as e:
            logger.warning(f"Could not save shard index: {e}")

        return shard_index

    def get_progress_info(self):
        return {
            "shard_index": self.current_shard_idx,
            "offset": self.current_offset,
            "total_shards": len(self.shard_files),
            "progress_percent": (self.current_shard_idx / len(self.shard_files)) * 100
        }

    def _load_shard(self, shard_path):
        try:
            return torch.load(shard_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Failed to load shard {shard_path}: {e}")
            raise

    def __iter__(self):
        for shard_idx in range(self.current_shard_idx, len(self.shard_files)):
            shard_path = self.shard_files[shard_idx]

            try:

                logger.debug(f"Loading shard {shard_idx}: {shard_path}")
                tokens = self._load_shard(shard_path)

                start_offset = self.current_offset if shard_idx == self.current_shard_idx else 0

                for i in range(start_offset, len(tokens) - self.context_length, self.context_length):
                    window = tokens[i:i + self.context_length + 1]

                    self.current_shard_idx = shard_idx
                    self.current_offset = i

                    yield window

                self.current_offset = 0

            except Exception as e:
                logger.error(f"Error processing shard {shard_path}: {e}")
                continue

        logger.info("Finished iterating through all shards")

    def __len__(self):

        total_tokens = len(self.shard_files) * self.shard_size
        return total_tokens // self.context_length

    def get_stats(self):
        return {
            "total_shard_files": len(self.shard_files),
            "estimated_total_tokens": len(self.shard_files) * self.shard_size,
            "context_length": self.context_length,
            "estimated_samples": len(self),
            "current_shard": self.current_shard_idx,
            "current_offset": self.current_offset,
            "shard_folder": str(self.shard_folder)
        }


class ShardManager:

    def __init__(self, max_cached_shards=2):
        self.max_cached_shards = max_cached_shards
        self.cache = {}
        self.access_order = []

    def get_shard(self, shard_path):
        str_path = str(shard_path)

        if str_path in self.cache:

            self.access_order.remove(str_path)
            self.access_order.append(str_path)
            return self.cache[str_path]

        tokens = torch.load(shard_path, map_location='cpu')

        if len(self.cache) >= self.max_cached_shards:

            lru_path = self.access_order.pop(0)
            del self.cache[lru_path]

        self.cache[str_path] = tokens
        self.access_order.append(str_path)

        return tokens

    def clear_cache(self):
        self.cache.clear()
        self.access_order.clear()
