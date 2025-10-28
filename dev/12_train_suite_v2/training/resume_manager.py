import os
import json
from pathlib import Path

class ResumeManager:

    def __init__(self, cfg):
        self.progress_file = Path(cfg['resume']['progress_file'])
        self.enabled = cfg['resume']['enabled']

    def load(self):
        if self.enabled and self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

    def save(self, progress: dict):
        if self.enabled:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
