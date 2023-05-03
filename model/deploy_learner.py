import time

from model.transformers.causal_lm_modeling import CausalLMModel
from model.vicuna.modeling import Vicuna
import os
import json
import logging
logging.basicConfig(level=logging.INFO)


def main():
    # model = Vicuna("eachadea/vicuna-7b-1.1", load_8bit=True, cache_dir='/root/autodl-tmp/vicuna/cache')
    model = CausalLMModel('chavinlo/alpaca-native', load_8bit=True, cache_dir='/root/autodl-tmp/alpaca/cache')
    exchange_info_path = "model/exchange/info.json"
    exchange_response_path = "model/exchange/response.json"
    while True:
        if os.path.exists(exchange_info_path):
            with open(exchange_info_path, "r") as f:
                info = json.load(f)

            if not info['done']:
                action = info['action']
                if action == 'learn':
                    logging.info("learn")
                    model.learn(info['instruction'], info['output'])
                elif action == 'chat':
                    logging.info("chat")
                    text = model.generate(info['instruction'])
                elif action == 'quit':
                    break

                # for all actions except 'quit', set 'done' to True
                info['done'] = True
                with open(exchange_info_path, 'w') as f:
                    json.dump(info, f)
                with open(exchange_response_path, 'w') as f:
                    json.dump({'text': text}, f)

        time.sleep(1)


if __name__ == "__main__":
    main()