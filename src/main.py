import threading
import time
import sys
import naive_bayes 
from data_handler import single_stock
import runtime_hanlder

def main_loop():
    while True:
        if runtime_hanlder.exit_event.is_set():
            print("Exit event detected. Stopping...")
            break
            
        if runtime_hanlder.restart_event.is_set():
            print("Restart event detected. Restarting...")
            runtime_hanlder.restart_event.clear()
            sys.stdin.flush()
            continue
            
        try:
            ticker_input = input("\nEnter stock ticker symbol (e.g., AAPL, MSFT) or 'q' to quit: ").strip().upper()
            
            if ticker_input.lower() == 'q':
                runtime_hanlder.exit_event.set()
                break
                
            if ticker_input.lower() == 'r':
                runtime_hanlder.restart_event.set()
                continue
                
            X, y = single_stock(ticker_input)
            
            result = naive_bayes.naive_bayes_classifier(X, y)
            
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting...")
            runtime_hanlder.exit_event.set()
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Continuing...")
            time.sleep(1)

if __name__ == "__main__":
    try:
        runtime_hanlder.start_listeners()
        main_loop()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    finally:
        runtime_hanlder.exit_event.set()
        sys.exit(0)