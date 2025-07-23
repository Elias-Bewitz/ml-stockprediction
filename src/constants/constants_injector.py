import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'scriptable_stats.json')
registry = {}

def _load_constants():
    with open(_CONFIG_PATH, 'r') as f:
        return json.load(f)

_constants = _load_constants()

__all__ = list(_constants.keys())

def __getattr__(name):
    try:
        return _constants[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name}")
    
def get_enabled_features():
    return [k for k, v in _constants.items() if v]

registry = get_enabled_features()

def reload_constants():
    global _constants, __all__, registry
    _constants = _load_constants()
    __all__ = list(_constants.keys())
    registry = get_enabled_features()