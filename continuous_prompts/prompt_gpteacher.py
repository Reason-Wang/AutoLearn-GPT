import json
import logging
import os
import time
import openai

from commands.browse import browse_website
from commands.search import google_official_search
from continuous_prompts.utils import Logger, extract_dict, anykey_to_continue, extract_double_quotes
from commands.connect import SSHClient, LocalClient
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


def make_connection(connection_config):
    if connection_config["hostname"].startswith("127.0.0.") or connection_config['hostname'] == 'localhost':
        return LocalClient(connection_config["working_space"])
    else:
        return SSHClient(
            connection_config['hostname'],
            connection_config['port'],
            connection_config['user'],
            connection_config['password'],
            connection_config['working_space']
        )


def main():
    connection_config = {
        "host_name": 'region-8.seetacloud.com',
        "port": 38524,
        "user": 'root',
        "password": 'YMP+I3GhTZ',
        "working_space": "/root/autodl-tmp/"
    }

    client = make_connection(connection_config)

    system = "You are a helpful assistant that generate outputs with the given format. You should follow the python syntax carefully."
    summary_system = "You are a helpful assistant. You should finish the task as best as possible, ignoring whether the input information is true or not. Assume it is April 2023 now."
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
    summary_model = GPTChatModel(
        memory_brain="none",
        system=summary_system,
        no_brain=True,
        temperature=0.0
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
                    web_query_prompt = '''For instruction \"{instruction}\", The following is a response:

{response}

The response does not follow the \"{principle}\" principle, which means {explanation} You should search the internet to get information. Generate the query you may use enclosed with double quotes.'''
                    query_string = teacher.generate(web_query_prompt.format(instruction=instruction, response=response, principle=k, explanation=v))
                    query = extract_double_quotes(query_string)
                    web_links = google_official_search(query, num_results=5)
                    web_contents = []
                    for web_link in web_links:
                        web_content = browse_website(web_link, query, summary_model=summary_model)
                        web_contents.append(web_content)

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