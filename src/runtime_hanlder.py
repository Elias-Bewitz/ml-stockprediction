import os
import sys
import json
import time
import threading
import functools
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_data       = {}                     # live config dict
_lock       = threading.Lock()       # guard writes to config file

exit_event    = threading.Event()    # signals "quit now"
restart_event = threading.Event()    # signals "restart now"

def _load_config():
    sys.stdin.flush()
    global _data
    try:
        with open(CONFIG_PATH, "r") as f:
            _data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _data = {}
    return _data

def _save_config(cfg: dict):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)

_load_config()

def register_feature(func):
    name = func.__name__
    cfg = _load_config()
    with _lock:
        if name not in cfg:
            cfg[name] = False
            _save_config(cfg)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def _key_listener():
    import msvcrt
    while True:
        if exit_event.is_set():
            break
        if msvcrt.kbhit():
            ch = msvcrt.getch().lower()
            # Only process keys when not in input mode
            # This is a simple check - in a real app you'd want more sophisticated handling
            if ch == b'r':
                print("\nDetected 'r' → restarting.")
                restart_event.set()
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage

def start_key_listener():
    t = threading.Thread(target=_key_listener, daemon=True)
    t.start()

class _ConfigChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if (not event.is_directory
            and os.path.abspath(event.src_path) == CONFIG_PATH):
            _load_config()
            sys.stdin.flush()
            print(f"🔄 config.json changed → reload and restarting.")
            restart_event.set()

def start_config_watcher():
    observer = Observer()
    observer.schedule(
        _ConfigChangeHandler(),
        path=os.path.dirname(CONFIG_PATH) or ".",
        recursive=False
    )
    observer.daemon = True
    observer.start()

def _monitor_events():
    while True:
        if exit_event.is_set():
            os._exit(0)
        if restart_event.is_set():
            os.execv(sys.executable, [sys.executable] + sys.argv)
        time.sleep(0.1)

def start_event_monitor():
    t = threading.Thread(target=_monitor_events, daemon=True)
    t.start()

def start_listeners():
    #start_key_listener()
    start_config_watcher()
    start_event_monitor()

def get_enabled_features():
    return [k for k, v in _data.items() if v]

registry = _data

def __getattr__(name: str):
    if name in _data:
        return _data[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(_data.keys())

__all__ = list(_data.keys()) + [
    "start_listeners", "register_feature",
    "get_enabled_features", "registry"
]