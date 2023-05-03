import json
import logging
import os
import time
import openai
from continuous_prompt.utils import Logger, extract_dict, anykey_to_continue
from commands.connect import SSHClient
from dotenv import load_dotenv
from termcolor import colored
from model.openai.modeling_chat import GPTChatModel

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"

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

    system = "You are a helpful assistant that generate outputs with the given format. You should follow the python syntax carefully."
    in_contexts = [['''Generate data for an old dog.

Format:
\'\'\'python
{
    "age": number # the age of the dog
    "body_length": number # the length from head to tail, measured in centimeters
    "is_male": True or False # whether the dog is male
    "description": "the description of the dog"
}
\'\'\'''', '''
Here is the data for the dog.
\'\'\'python
{
    "age": 8,
    "body_length: 65,
    "is_male": False,
    "description": "It is a cute, old dog with spots covered on its body."
}
\'\'\'''']]
    teacher = GPTChatModel(
        memory_brain="none",
        system=system,
        no_brain=True,
        temperature=0.0,
        incontext_example=in_contexts
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

Format:
\'\'\'python
{{
    "principle name 1": "Explanation for the principle 1",
    "principle name 2": "Explanation for the principle 2",
    # Other principles with the same format
}}
\'\'\''''
        LOGGER.info(principles_prompt.format(instruction=instruction), "System", "blue")
        anykey_to_continue()
        principles_dict_string = teacher.generate(principles_prompt.format(instruction=instruction))
        LOGGER.info(principles_dict_string, "Model", "yellow")
        principles_dict = extract_dict(principles_dict_string)

        should_learn = False
        for k, v in principles_dict.items():
            principle_check_prompt = '''For instruction "{instruction}" The response is:
            
{response}


"{principle}" is a principle meaning that {explanation} Find whether the response followed the principle.

Format:
\'\'\'python
{{
    "follow_principle": True or False, # a boolean value indicates whether the response follows the principle
    "explanation": "Explain why the response follows or does not follow the principle"
}}
\'\'\''''
            LOGGER.info(principle_check_prompt.format(instruction=instruction, principle=k, explanation=v.lower(), response=response), "System", "blue")
            anykey_to_continue()
            principle_check_dict_string = teacher.generate(principle_check_prompt.format(instruction=instruction, principle=k, explanation=v.lower(), response=response))
            LOGGER.info(principle_check_dict_string, "Model", "yellow")
            principle_check_dict = extract_dict(principle_check_dict_string)
            if not principle_check_dict["follow_principle"]:
                should_learn = True
                action_prompt = '''For instruction \"{instruction}\", The following is a response:
{response}

The response does not follow the \"{principle}\" principle, which means {explanation} Choose one of the actions to do:
1. Search the internet. Suitable for collecting information, facts verification, etc.
2. Rewrite the response.

Format:
\'\'\'python
{{
    "action_number": 1 or 2, # the number corresponding to the action
    "explanation": "Explain why to choose the action."
}}\'\'\''''
                LOGGER.info(action_prompt.format(instruction=instruction, response=response, principle=k, explanation=v), "System", "blue")
                anykey_to_continue()
                action_dict_string = teacher.generate(action_prompt.format(instruction=instruction, response=response, principle=k, explanation=v))
                LOGGER.info(action_dict_string, "Model", "yellow")
                action_dict = extract_dict(action_dict_string)
                if action_dict['action_number'] == 1:
                    raise NotImplementedError
                elif action_dict['action_number'] == 2:
                    rewrite_prompt = '''For instruction \"{instruction}\", The following is a response:
                    
{response}

The response does not follow the \"{principle}\" principle, which means {explanation}. Rewrite the response to make it follow the principle.'''
                    LOGGER.info(rewrite_prompt.format(instruction=instruction, response=response, principle=k, explanation=v), "System", "blue")
                    anykey_to_continue()
                    new_response = teacher.generate(rewrite_prompt.format(instruction=instruction, response=response, principle=k, explanation=v))
                    LOGGER.info(new_response, "Model", "yellow")
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
            LOGGER.info("Instructed the learner to learn...", "System", "blue")
            anykey_to_continue()














if __name__ == "__main__":
    main()