import time
from model.vicuna.modeling import Vicuna
import os
import json
import logging
logging.basicConfig(level=logging.INFO)


def main():
    model = Vicuna("eachadea/vicuna-13b-1.1", load_8bit=True)
    exchange_info_path = "model/exchange/info.json"
    while True:
        if os.path.exists(exchange_info_path):
            with open(exchange_info_path, "r") as f:
                info = json.load(f)

            action = info['action']
            if action == 'learn':
                logging.info("learn")
            elif action == 'chat':
                logging.info("chat")
            elif action == 'quit':
                break
        time.sleep(1)


if __name__ == "__main__":
    main()