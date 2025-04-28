from stable_baselines3 import PPO
import os, glob, re
import numpy as np
from config import MODEL_PATH

_model = None

def resolve_model_path():
    """Return existing MODEL_PATH or fallback to latest RL2 chunk."""
    default = MODEL_PATH
    if default and os.path.exists(default):
        return default
    model_dir = os.path.dirname(default) if default else 'models'
    pattern = os.path.join(model_dir, 'rl2_multitask_chunk_*.zip')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No model file found at {default} nor RL2 chunks in {model_dir}")
    # parse chunk index
    def idx(f):
        m = re.search(r'chunk_(\d+)\.zip$', f)
        return int(m.group(1)) if m else -1
    files.sort(key=idx)
    return files[-1]


def load_model(path: str = None) -> PPO:
    """Load and cache the RL model, resolving path if needed."""
    global _model
    if _model is None:
        model_path = path or resolve_model_path()
        _model = PPO.load(model_path)
    return _model


def predict_action(features):
    """Infer action: 0=HOLD,1=BUY_CE,2=BUY_PE"""
    model = load_model()
    action, _states = model.predict(features, deterministic=True)
    # Ensure action is Python int (not numpy.ndarray)
    if hasattr(action, 'item'):
        action = action.item()
    return {0: "HOLD", 1: "BUY_CE", 2: "BUY_PE"}.get(action, "HOLD")
