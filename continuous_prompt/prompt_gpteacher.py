import json
import os
import time

import openai
from continuous_prompt.utils import Logger
from model.vicuna.modeling import Vicuna
from commands.connect import SSHClient
from dotenv import load_dotenv
from termcolor import colored
from model.openai.modeling_chat import GPTChatModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TABLE_NAME = os.getenv("TABLE_NAME", "")

openai.api_key = OPENAI_API_KEY

LOGGER = Logger()


def main():
    hostname = 'connect.neimeng.seetacloud.com'
    port = 22974
    user = 'root'
    password = 'hGREYe0Z3c'
    working_space = "/root/autodl-tmp/"
    client = SSHClient(hostname, port, user, password, working_space)

    system = "You are a helpful and curious assistant that always " \
             "want to know the correct answers of questions and " \
             "find the most efficient way to do that."

    teacher = GPTChatModel(
        memory_brain="none",
        system=system,
        no_brain=True
    )

    info = {
        "done": True,
        "action": "none",
    }

    while True:
        LOGGER.info("Input an instruction \"quit\" to quit: ", "System", "blue")
        instruction = input()

        if instruction == "quit":
            break

        info["instruction"] = instruction
        info["done"] = False
        info["action"] = "generate"
        with open("model/exchange/info.json", "w") as f:
            json.dump(info, f)
        client.send_file("model/exchange/info.json")

        # waiting for response
        while True:
            client.get_file("model/exchange/info.json")
            with open("model/exchange/info.json", 'r') as f:
                info = json.load(f)
            if info["done"]:
                break
            time.sleep(0.5)

        client.get_file("model/exchange/response.json")
        with open("model/exchange/response.json", 'r') as f:
            response = json.load(f)











if __name__ == "__main__":
    main()