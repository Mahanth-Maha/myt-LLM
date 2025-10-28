import os
from dotenv import load_dotenv
load_dotenv() 

import logging
from logging import Logger, FileHandler, StreamHandler, Formatter

def setup_logging(cfg, name = __name__):

    log_cfg = cfg.get('logging', {})
    log_dir = log_cfg.get('log_files_dir', './checkpoints/logs')
    os.makedirs(log_dir, exist_ok=True)
    
    file_map = {
        'train': log_cfg.get('train_log_file'),
        'dry_run': log_cfg.get('dry_run_log_file'),
        'samples': log_cfg.get('samples_log_file')
    }
    file_path = file_map.get(name, os.path.join(log_dir, f"{name}.log"))
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        return logger
    
    fmt = Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    
    fh = FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    ch = StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger
