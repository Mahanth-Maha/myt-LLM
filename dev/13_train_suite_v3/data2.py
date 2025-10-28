import os
import json
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader


DEFAULT_AVG_SHARD_SIZE = 750_000_000

class MultiDirectoryStreamingTextDataset(IterableDataset):
    def __init__(
        self,
        directory_list,
        context_length,
        resume_info = None,
        exclude_dirs = None,
        max_cached_shards = 2
    ):
        self.directory_list = [Path(d) for d in directory_list]
        self.context_length = context_length
        self.resume_info = resume_info or {}
        self.exclude_dirs = exclude_dirs or ['old_shards']

        self.shard_manager = ShardManager(max_cached_shards=max_cached_shards)

        self.ordered_shards = self._discover_ordered_shards()
        self.metadata_cache = self._load_all_metadata()

        self.total_tokens = self._calculate_total_tokens()

        self.current_shard_idx = self.resume_info.get('shard_index', 0)
        self.current_offset = self.resume_info.get('offset', 0)

        self._log_dataset_info()

    def _discover_ordered_shards(self):
        ordered_shards = []

        for dir_idx, directory in enumerate(self.directory_list):
            if not directory.exists():
                print(f"⚠️ Directory not found: {directory}")
                continue

            dir_shards = self._discover_shards_in_directory(directory, dir_idx)
            ordered_shards.extend(dir_shards)

        print(f"Discovered {len(ordered_shards)} total shards across {len(self.directory_list)} directories")
        return ordered_shards

    def _discover_shards_in_directory(self, directory, dir_idx):
        shard_info_list = []
        
        for shard_file in directory.rglob("shard_*.pt"):
            if not shard_file.is_file():
                continue
            
            if any(exclude_dir in str(shard_file.parent) for exclude_dir in self.exclude_dirs):
                continue

            relative_path = shard_file.relative_to(directory)
            sort_key = str(relative_path).replace('/', '_').replace('\\', '_')

            shard_info = {
                'path': shard_file,
                'directory_index': dir_idx,
                'directory_name': directory.name,
                'relative_path': str(relative_path),
                'sort_key': sort_key,
                'metadata_dir': self._find_metadata_dir(shard_file)
            }
            shard_info_list.append(shard_info)
        
        shard_info_list.sort(key=lambda x: x['sort_key'])

        print(f"📁 Found {len(shard_info_list)} shards in {directory.name}")
        return shard_info_list

    def _find_metadata_dir(self, shard_file):
        current_dir = shard_file.parent
        while current_dir != current_dir.parent:
            metadata_file = current_dir / "metadata.json"
            if metadata_file.exists():
                return current_dir
            current_dir = current_dir.parent

        return None

    def _load_all_metadata(self):
        metadata_cache = {}

        for shard_info in self.ordered_shards:
            metadata_dir = shard_info['metadata_dir']
            if metadata_dir and str(metadata_dir) not in metadata_cache:
                try:
                    metadata_file = metadata_dir / "metadata.json"
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata_cache[str(metadata_dir)] = metadata
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️ Could not load metadata from {metadata_file}: {e}")
                    metadata_cache[str(metadata_dir)] = {}

        return metadata_cache

    def _calculate_total_tokens(self):
        total = 0

        for shard_info in self.ordered_shards:
            metadata_dir = shard_info['metadata_dir']
            if metadata_dir:
                metadata = self.metadata_cache.get(str(metadata_dir), {})
                shard_filename = shard_info['path'].name

                file_counts = metadata.get('file_token_counts', {})
                if shard_filename in file_counts:
                    total += file_counts[shard_filename]
                else:                    
                    total_tokens = metadata.get('total_tokens', DEFAULT_AVG_SHARD_SIZE)  
                    total_files = metadata.get('total_pt_files', 1)
                    total += total_tokens // total_files
            else:
                total += DEFAULT_AVG_SHARD_SIZE  

        return total

    def _log_dataset_info(self):
        print(f"📦 Dataset initialized with {len(self.ordered_shards)} shards")
        print(f"\tTotal estimated tokens: {self.total_tokens:,}")
        print(f"\tContext length: {self.context_length}")
        print(f"\tEstimated samples: {self.total_tokens // self.context_length:,}")

        if self.resume_info:
            print(f"⏯️ Resuming from shard {self.current_shard_idx}, offset {self.current_offset}")

        dir_summary = {}
        for shard_info in self.ordered_shards:
            dir_name = shard_info['directory_name']
            dir_summary[dir_name] = dir_summary.get(dir_name, 0) + 1

        for dir_name, count in dir_summary.items():
            print(f"  🔹 {dir_name}:  {count} shards")

    def get_progress_info(self):
        return {
            "shard_index": self.current_shard_idx,
            "offset": self.current_offset,
            "total_shards": len(self.ordered_shards),
            "progress_percent": (self.current_shard_idx / len(self.ordered_shards)) * 100 if self.ordered_shards else 0,
            "current_directory": self.ordered_shards[self.current_shard_idx]['directory_name'] if self.current_shard_idx < len(self.ordered_shards) else "finished",
            "current_shard_path": str(self.ordered_shards[self.current_shard_idx]['path']) if self.current_shard_idx < len(self.ordered_shards) else "finished"
        }

    def get_validation_shard_info(self):
        if not self.ordered_shards:
            return {}

        last_shard = self.ordered_shards[-1]
        metadata_dir = last_shard['metadata_dir']

        tokens_in_shard = DEFAULT_AVG_SHARD_SIZE
        if metadata_dir:
            metadata = self.metadata_cache.get(str(metadata_dir), {})
            file_counts = metadata.get('file_token_counts', {})
            shard_filename = last_shard['path'].name
            if shard_filename in file_counts:
                tokens_in_shard = file_counts[shard_filename]

        return {
            'path': last_shard['path'],
            'tokens': tokens_in_shard,
            'samples': max(0, tokens_in_shard - self.context_length),
            'directory': last_shard['directory_name']
        }

    def __iter__(self):
        """Iterate through dataset, yielding context windows."""
        for shard_idx in range(self.current_shard_idx, len(self.ordered_shards)):
            shard_info = self.ordered_shards[shard_idx]
            shard_path = shard_info['path']

            try:
                tokens = self.shard_manager.get_shard(shard_path)
                start_offset = self.current_offset if shard_idx == self.current_shard_idx else 0
                for i in range(start_offset, len(tokens) - self.context_length, self.context_length):
                    window = tokens[i:i + self.context_length + 1]
                    self.current_shard_idx = shard_idx
                    self.current_offset = i
                    yield window[:-1], window[1:]
                self.current_offset = 0

            except Exception as e:
                print(f"❓ Error processing shard {shard_path}: {e}")
                continue

        print("✅ Finished iterating through all shards")

    def __len__(self):
        return max(1, self.total_tokens // self.context_length)

    def get_stats(self):
        return {
            "total_directories": len(self.directory_list),
            "total_shard_files": len(self.ordered_shards),
            "total_tokens": self.total_tokens,
            "context_length": self.context_length,
            "estimated_samples": len(self),
            "current_shard": self.current_shard_idx,
            "current_offset": self.current_offset,
            "directories": [str(d) for d in self.directory_list],
            "exclude_dirs": self.exclude_dirs
        }


class ShardManager:
    def __init__(self, max_cached_shards = 2):
        self.max_cached_shards = max_cached_shards
        self.cache = OrderedDict()

    def get_shard(self, shard_path):
        str_path = str(shard_path)

        if str_path in self.cache:
            self.cache.move_to_end(str_path)
            return self.cache[str_path]

        try:
            tokens = torch.load(shard_path, map_location='cpu')
        except Exception as e:
            print(f"❌ Failed to load shard {shard_path}: {e}")
            raise

        while len(self.cache) >= self.max_cached_shards:
            self.cache.popitem(last=False)

        self.cache[str_path] = tokens

        return tokens

    def clear_cache(self):
        self.cache.clear()

    def get_cache_info(self):
        return {
            "cached_shards": len(self.cache),
            "max_cache_size": self.max_cached_shards,
            "cached_paths": list(self.cache.keys())
        }


class RobustValidationDataset(Dataset):
    def __init__(
        self,
        dataset_manager: MultiDirectoryStreamingTextDataset,
        min_samples=1000,
        max_cycles = 10
    ):
        self.dataset_manager = dataset_manager
        self.context_length = dataset_manager.context_length
        self.min_samples = min_samples
        self.max_cycles = max_cycles

        self.val_shard_info = dataset_manager.get_validation_shard_info()

        if not self.val_shard_info:
            raise ValueError("No validation shard available")

        self.tokens = dataset_manager.shard_manager.get_shard(self.val_shard_info['path'])
        self.samples_per_cycle = max(0, len(self.tokens) - self.context_length)
        self.num_cycles = min(max_cycles, max(1, min_samples // max(1, self.samples_per_cycle)))
        self.total_samples = self.samples_per_cycle * self.num_cycles

        print(f"🔍 Validation dataset: {self.samples_per_cycle} samples/cycle × {self.num_cycles} cycles = {self.total_samples} total samples")
        print(f"📁 Using validation shard: {self.val_shard_info['directory']}/{self.val_shard_info['path'].name}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cycle_idx = idx // self.samples_per_cycle
        pos_in_cycle = idx % self.samples_per_cycle

        offset = (cycle_idx * 100) % max(1, self.samples_per_cycle // 4)
        actual_pos = (pos_in_cycle + offset) % self.samples_per_cycle

        start_idx = actual_pos
        end_idx = start_idx + self.context_length + 1

        if end_idx > len(self.tokens):
            start_idx = len(self.tokens) - self.context_length - 1
            end_idx = len(self.tokens)

        window = self.tokens[start_idx:end_idx]
        return window[:-1], window[1:]


class ValidationDatasetStreamer:
    def __init__(
        self,
        directory_list,
        context_length,
        exclude_dirs = None,
        min_val_samples = 1000,
        max_val_cycles = 10
    ):
        self.dataset_manager = MultiDirectoryStreamingTextDataset(
            directory_list=directory_list,
            context_length=context_length,
            exclude_dirs=exclude_dirs or ['old_shards']
        )

        self.min_val_samples = min_val_samples
        self.max_val_cycles = max_val_cycles

    def get_val_loader(
        self,
        batch_size,
        num_workers = 0,
        pin_memory = False,
        shuffle = False
    ) :
        val_dataset = RobustValidationDataset(
            self.dataset_manager,
            min_samples=self.min_val_samples,
            max_cycles=self.max_val_cycles
        )

        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


class StreamingTextDataset(MultiDirectoryStreamingTextDataset):
    def __init__(
        self,
        shard_folder,
        context_length,
        shard_size = DEFAULT_AVG_SHARD_SIZE,
        resume_info = None
    ):
        directory_list = [shard_folder]
        super().__init__(
            directory_list=directory_list,
            context_length=context_length,
            resume_info=resume_info
        )
