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
        logging.error("Syntax error, try to fix string.")
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


def anykey_to_continue():
    input("\nContinue\n")