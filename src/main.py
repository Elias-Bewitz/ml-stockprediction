import os
import threading
import time
import sys
import naive_bayes 
from data_handler import single_stock
import input_event_on_thread
import constants
import logger

def main():
    try:
        watcher.pause()   # pause background watcher before input()
        ticker_input = input(
            "\nEnter stock ticker symbol (e.g., AAPL, MSFT) or 'q' to quit: "
        ).strip()
        watcher.resume()  # resume watching immediately after

        cmd = ticker_input.lower()
        if cmd == 'q':
            watcher.send_event_q()
            return                   # ⇐ don’t go on to lookup “Q”
        elif cmd == 'r':
            watcher.send_event_r()
            return                   # ⇐ don’t go on to lookup “R”

        # only runs when you’ve got a real ticker
        X, y = single_stock(ticker_input.upper())
        result = naive_bayes.naive_bayes_classifier(X, y)
        logger.log_metrics(result)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    # 1) start the watcher
    watcher = input_event_on_thread.KeyWatcher()
    watcher.start()

    # 2) run your first query loop
    main()

    # 3) now wait for q/r events
    while True:
        event = watcher.get_event(block=True)  
        if event == 'q':
            watcher.stop()
            print("Exiting…")
            sys.exit(0)

        elif event == 'r':
            watcher.stop()                   # tear down old thread
            constants.reload_constants()     # pick up new constants.json
            # restart watcher so it’ll capture keys again
            watcher = input_event_on_thread.KeyWatcher()
            watcher.start()
            # restart your main logic
            main()