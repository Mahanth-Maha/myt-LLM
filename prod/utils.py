import os
import torch
import yaml
import pytz

from architecture import DecoderOnlyTransformer

def convert2hr(num):
    for unit in ['','K','M','B']:
        if abs(num) < 1000:
            return f"{num:.2f} {unit}"
        num /= 1000.0
    return f"{num:.2f} T"

def convert2hr1(n):
    n = float(n)
    for u in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.1f} {u}"
        n /= 1000.0
    return f"{n:.1f} P"


def fmt_time(total_seconds):
    total_seconds = float(total_seconds)
    neg = total_seconds < 0
    total_seconds = abs(total_seconds)
    d = int(total_seconds // 86400)
    h = int((total_seconds % 86400) // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{'-' if neg else ''}{d}d {h:02d}h {m:02d}m {s:02d}s"

def fmt_dt_ist(dt):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    ist = pytz.timezone("Asia/Kolkata")
    dt = dt.astimezone(ist)
    return f"{months[dt.month-1]} {dt.day:02d} {dt.year} {days[dt.weekday()]} {dt:%H:%M:%S}"



def get_time_str(time_in_secs):
    neg = False
    if time_in_secs <0:
        time_in_secs = abs(time_in_secs)
        neg = True
    days = int(time_in_secs // 86400)
    hours = int((time_in_secs % 86400) // 3600)
    minutes = int((time_in_secs % 3600) // 60)
    seconds = int(time_in_secs % 60)
    return f"{'-' if neg else ''} {days} Days {hours:2d} Hours {minutes:2d} Mins {seconds} Secs"


def load_config_from_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_configs(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def overrides_to_dict(args):
    result = {}
    for key, value in vars(args).items():
        if value is None or key == "config_file":
            continue
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def get_model(model_dir, device='cpu', model_kwargs=None, config_path=None, default_config_path=None,verbose = False):
    model_cfg_path = os.path.join(model_dir, "model_config.pt")
    model_weights_path = os.path.join(model_dir, "model.pt")

    if not os.path.isfile(model_weights_path):
        raise FileNotFoundError(f"☹️ Model weights not found at: {model_weights_path}")

    if os.path.isfile(model_cfg_path):
        try:
            loaded_kwargs = torch.load(model_cfg_path, map_location="cpu")
            if isinstance(loaded_kwargs, dict):
                model_kwargs = loaded_kwargs
                if verbose:
                    print(f"😉 Loaded model kwargs from {model_cfg_path}")
        except Exception as e:
            print(f"⚠️ Could not load model_config.pt: {e}")

    if model_kwargs is None and config_path is not None:
        try:
            base_cfg = {}
            if default_config_path and os.path.isfile(default_config_path):
                base_cfg = load_config_from_yaml(default_config_path)
            config = load_config_from_yaml(default_config_path)
            merged_cfg = merge_configs(base_cfg, config)
            m_cfg = merged_cfg.get("model", merged_cfg)

            model_kwargs = dict(
                vocab_size=m_cfg["vocab_size"],
                context_length=m_cfg["context_length"],
                model_dimension=m_cfg["dimension"],
                n_heads=m_cfg["n_heads"],
                Nx_blocks=m_cfg["num_layers"],
                ffn_hid_dim=m_cfg["hidden_dimension"],
                n_kv_heads=m_cfg["n_kv_heads"] if m_cfg["n_kv_heads"] != -1 else None,
                dropout=m_cfg.get("dropout", 0.0),
                tie_weights=m_cfg.get("tie_weights", True),
                use_checkpoint=m_cfg.get("use_checkpoint", False),
                checkpoint_ratio=m_cfg.get("checkpoint_ratio", 0.5),
            )
            if verbose:
                print(f"😉 Built model kwargs from config.")
        except Exception as e:
            print(f"⚠️ Failed to build model_kwargs from config: {e}")
            
    if model_kwargs is None:
        print(f"⚠️ Falling back to hardcoded default model kwargs.")
        model_kwargs = dict(
            vocab_size=100352,
            context_length=128,
            model_dimension=128,
            n_heads=8,
            Nx_blocks=12,
            ffn_hid_dim=352,
            n_kv_heads=None,
            dropout=0.0,
            tie_weights=True,
            use_checkpoint=False,
            checkpoint_ratio=1.0,
        )
    model = DecoderOnlyTransformer(**model_kwargs)

    try:
        state_dict = torch.load(model_weights_path, map_location="cpu")

        for prefix in ["_orig_mod.", "module."]:
            if any(k.startswith(prefix) for k in state_dict.keys()):
                print(f"⚠️ Detected '{prefix}' prefix in checkpoint — stripping it.")
                state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}


        model.load_state_dict(state_dict)
        model.to(device).eval()
        if verbose:
            print(f"😉 Model loaded successfully on {device}")
    except Exception as e:
        raise RuntimeError(f"☹️ Failed to load model weights: {e}")

    return model