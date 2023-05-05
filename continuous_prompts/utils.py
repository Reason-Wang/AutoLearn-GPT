import os
import re
import ast
import logging
from termcolor import colored
from model.openai.modeling_chat import GPTChatModel


def extract_dict(string_with_dict: str, llm_fix=True):
    # For markdown
    try:
        dict_string = re.search("```[\s\S]*```", string_with_dict).group()[3:-3]
    except Exception as e:
        logging.debug("May not be the markdown syntax, try general brace.")
        dict_string = re.search("{[\s\S]*}", string_with_dict).group()

    if dict_string.startswith("python"):
        dict_string = dict_string[6:]

    # TODO: probably bug here
    possible_dict_string = None
    try:
        extracted_dict = ast.literal_eval(dict_string)
    except (SyntaxError, ValueError) as e:
        logging.debug(f"{e} for \"{dict_string}\", try to fix string.")
        def replace(match_obj):
            if match_obj.group(1).endswith(','):
                replace_holder = match_obj.group(1)[:-1]
            else:
                replace_holder = match_obj.group(1)
            if replace_holder.isdigit():
                return '": ' + replace_holder + ', "'
            elif replace_holder == 'true' or replace_holder == 'false':
                replace_holder = replace_holder[0].upper() + replace_holder[1:]
                return '": ' + replace_holder + ', "'
            else:
                if replace_holder.startswith('"') and replace_holder.endswith('"'):
                    return '": ' + replace_holder + ','
                return '": ' + r'"' + replace_holder + '",'

        possible_dict_string = re.sub(r'":\s*([\s\S]*?)\n\s*["|}]', replace, dict_string)
        if possible_dict_string.endswith(','):
            possible_dict_string = possible_dict_string[:-1] + "\n}"
        elif possible_dict_string.endswith(r', "'):
            possible_dict_string = possible_dict_string[:-3] + "\n}"
    except Exception as e:
        logging.error(f"Error for \"{dict_string}\"")
        raise e

    if possible_dict_string is not None:
        logging.debug(f"Dict string after fixing: \"{possible_dict_string}\"")
        try:
            extracted_dict = ast.literal_eval(possible_dict_string)
        except SyntaxError as e:
            if llm_fix:
                logging.debug(f"Still syntax error, try to fix with llm.")
                model = GPTChatModel(memory_brain="", system="You are a helpful assistant. You should finish the task as best as possible.", no_brain=True)
                dict_fix_prompt = f"The following is a python dictionary string that may contain some syntax errors. For example, the first character of boolean values may be lower case, string may not be quoted. Try to fix it and generate the correct python dictionary string. \n{dict_string}"
                llm_fixed_dict_string = model.generate(dict_fix_prompt)
                logging.debug(f"Dict string after llm fixing: \"{llm_fixed_dict_string}\"")
                extracted_dict = extract_dict(llm_fixed_dict_string, llm_fix=False)
            else:
                raise e

    return extracted_dict


def extract_double_quotes(string_with_double_quotes: str):
    try:
        quoted_strings = re.findall("\"([^\"]*)\"", string_with_double_quotes)
        quoted_string = quoted_strings[0]
    except Exception as e:
        logging.error(f"Extract double quoted string error for {string_with_double_quotes}")
        raise e

    return quoted_string


def filter_unrelated_contents(contents, question, summary_model):
    filter_prompt = "The following is a text:\n\n{content}. " \
                    "Is it related to \"{question}\"\n\n " \
                    "You should respond with a python dictionary, which contains the following keys:\n" \
                    "\"related\": a boolean variable indicates whether the two are related. " \
                    "It should be False if the text states no information is given or the two are not related.\n" \
                    "\"explanation\": a string that explain why the \"related\" is True or False. " \
                    "Respond with only a python dictionary. Do not generate other python codes. Be careful to follow python syntax."

    filter_dicts = []
    for content in contents:
        dict = summary_model.generate(filter_prompt.format(content=content, question=question))
        dict = extract_dict(dict)
        filter_dicts.append(dict)
    return filter_dicts


def format_web_summary(web_contents):
    summary = ""
    for i, c in enumerate(web_contents):
        summary += f"Summary {i+1}: {c}\n"
    return summary


def anykey_to_continue():
    input(colored("\n### CONTINUE ###", "green"))


class Logger:
    def __init__(self):
        from logging import getLogger, INFO, StreamHandler, Formatter
        logger = getLogger()
        logger.setLevel(INFO)
        logger.handlers = []
        handler = StreamHandler()
        handler.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler)
        self.logger = logger

    def info(self, message, role, color):
        self.logger.info(colored(f"{role}: {message}", color))

    def debug(self, message, role, color):
        self.logger.debug(colored(f"{role}: {message}", color))


def use_proxy(http_port=1081):
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:" + str(http_port)
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:" + str(http_port)