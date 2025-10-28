import os
import tempfile
import shutil
import torch
import logging
import pytest
from data.streaming_dataset import StreamingTextDataset


log_dir = "./checkpoints/logs/tests"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "test_dataset.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)



@pytest.fixture(scope="module")
def dummy_shards():
    root = tempfile.mkdtemp()
    sub = os.path.join(root, "part-000")
    os.makedirs(sub, exist_ok=True)
    for i in range(2):
        tokens = torch.arange(0, 2000, dtype=torch.long)
        path = os.path.join(sub, f"shard_{i:05d}.pt")
        torch.save(tokens, path)
    yield root
    shutil.rmtree(root)

def test_shard_discovery_and_indexing(dummy_shards):
    ds = StreamingTextDataset(shard_folder=dummy_shards, context_length=128, shard_size=2000)
    assert len(ds.shard_files) == 2
    index_path = os.path.join(dummy_shards, "shard_index.json")
    assert os.path.isfile(index_path)
    info = ds.get_stats()
    assert info["total_shard_files"] == 2
    assert info["estimated_total_tokens"] == 4000
    assert info["context_length"] == 128
    assert info["estimated_samples"] == 31

def test_iteration_yields_windows(dummy_shards):
    ds = StreamingTextDataset(shard_folder=dummy_shards, context_length=128, shard_size=2000)
    iterator = iter(ds)
    first = next(iterator)
    assert first.shape[0] == 129
    for window in ds:
        assert isinstance(window, torch.Tensor)
        assert window.dtype == torch.long

def test_resume_functionality(dummy_shards):
    resume_info = {"shard_index": 1, "offset": 64}
    ds = StreamingTextDataset(shard_folder=dummy_shards, context_length=128, shard_size=2000, resume_info=resume_info)
    stats = ds.get_progress_info()
    assert stats["shard_index"] == 1
    assert stats["offset"] == 64
    window = next(iter(ds))
    assert window[0].item() == 64

if __name__ == "__main__":
    pytest.main([__file__, "-q"])
