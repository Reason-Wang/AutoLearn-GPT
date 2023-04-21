import re
import ast
import logging

def extract_dict(string_with_dict: str):
    # For markdown
    try:
        dict_string = re.search("```[\s\S]*```", string_with_dict).group()[3:-3]
    except Exception as e:
        logging.error("May not be the markdown syntax, try general brace.")
        dict_string = re.search("{[\s\S]*}", string_with_dict).group()

    if dict_string.startswith("python"):
        dict_string = dict_string[6:]

    # TODO: probably bug here
    try:
        extracted_dict = ast.literal_eval(dict_string)
    except SyntaxError as e:
        logging.error(f"Syntax error for \"{dict_string}\", try to fix string.")
        def replace(match_obj):
            if match_obj.group(1).endswith(','):
                replace_holder = match_obj.group(1)[:-1]
            else:
                replace_holder = match_obj.group(1)
            if replace_holder.isdigit():
                return r'": ' + replace_holder + r', "'
            else:
                return r'": ' + r'"' + replace_holder + r'",'

        possible_dict_string = re.sub(r'":\s*([\s\S]*?)\n\s*["|}]', replace, dict_string)
        if possible_dict_string.endswith(','):
            possible_dict_string = possible_dict_string[:-1] + "\n}"
        elif possible_dict_string.endswith(r', "'):
            possible_dict_string = possible_dict_string[:-3] + "\n}"

        extracted_dict = ast.literal_eval(possible_dict_string)
        logging.info("Fixed error.")

    return extracted_dict


def extract_double_quotes(string_with_double_quotes: str):
    quoted_strings = re.findall("\"([^\"]*)\"", string_with_double_quotes)
    return quoted_strings


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
        summary += f"Summary {i}: {c}\n\n"
    return summary


def anykey_to_continue():
    input("\nContinue\n")