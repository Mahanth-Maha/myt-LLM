import yaml
from box import Box

import os
from dotenv import load_dotenv
load_dotenv() 

root_dir = os.getenv("MYTLLM_ROOT")
default_config = os.getenv("MYTLLM_CONFIG")
default_config_path = os.path.join(root_dir,default_config)

def load_config(override_path= None, default_path= os.path.join(root_dir,default_config)):
    with open(default_path, "r") as f:
        cfg = yaml.safe_load

    if override_path:
        with open(override_path, "r") as f:
            override = yaml.safe_load(f)
        cfg = deep_update(cfg, override)

    return Box(cfg)

def deep_update(base, override) :
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base
