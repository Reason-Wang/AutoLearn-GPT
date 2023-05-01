import json
import logging
import os
import time
import openai
from continuous_prompt.utils import Logger, extract_dict
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
    hostname = 'region-8.seetacloud.com'
    port = 38524
    user = 'root'
    password = 'YMP+I3GhTZ'
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
        info["action"] = "chat"
        with open("model/exchange/info.json", "w") as f:
            json.dump(info, f)
        client.send_file("model/exchange/info.json")

        # waiting for response
        logging.info("waiting for response")
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

        logging.info(response['text'])
        response = response['text']

        principles_prompt = '''A good response for an instruction aims to solve the instruction perfectly. For the instruction \"{instruction}\", generate some principles that a good response should have.

You should generate a python dictionary with the following format:
{
    "principle name 1": "Explanation for the principle 1",
    "principle name 2": "Explanation for the principle 2",
    # Other principles with the same format
}'''
        principles_dict_string = teacher.generate(principles_prompt.format(instruction=instruction))
        principles_dict = extract_dict(principles_dict_string)

        should_learn = False
        for k, v in principles_dict.items():
            principle_check_prompt = '''For instruction "{instruction}" "{principle}" is a principle meaning that {explanation} Does the following response follow the principle?

{response}

You should generate a python dictionary with the following format:
{
    "follow_principle": True or False, # a boolean value indicates whether the response follows the principle
    "explanation": "Explain why the response follows or does not follow the principle"
}'''
            principle_check_dict_string = teacher.generate(principle_check_prompt.format(instruction=instruction, principle=k, explanation=v.lower(), response=response))
            principle_check_dict = extract_dict(principle_check_dict_string)
            if not principle_check_dict["follow_principle"]:
                should_learn = True
                rewrite_prompt = '''Rewrite a response based on the original response, which follows the principle.'''
                new_response = teacher.chat(rewrite_prompt)
                response = new_response

            teacher.clear_history()

        if should_learn:
            info["done"] = False
            info["action"] = "learn"
            info["instruction"] = instruction
            info["output"] = response
            with open("model/exchange/info.json", "w") as f:
                json.dump(info, f)
            client.send_file("model/exchange/info.json")














if __name__ == "__main__":
    main()