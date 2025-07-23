import threading
import queue
import sys
import os
import time
import msvcrt

class KeyWatcher:
    def send_event_q(self):
        self._event_queue.put('q')

    def send_event_r(self):
        self._event_queue.put('r')
        
    def __init__(self):
        self._event_queue = queue.Queue()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def get_event(self, block=False, timeout=None):
        try:
            return self._event_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        self._run_windows()

    def _run_windows(self):
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                time.sleep(0.1)
                continue
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    key = ch.decode().lower()
                except UnicodeDecodeError:
                    continue
                if key in ('q', 'r'):
                    self._event_queue.put(key)
            time.sleep(0.05)