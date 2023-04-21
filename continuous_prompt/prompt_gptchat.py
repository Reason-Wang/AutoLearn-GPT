import logging
import os

import pinecone
from dotenv import load_dotenv
from commands.browse import browse_website
from commands.search import google_official_search
from continuous_prompt.utils import (
    extract_dict,
    extract_double_quotes,
    anykey_to_continue,
    filter_unrelated_contents,
    format_web_summary
)
from model.openai.modeling_chat import GPTChatModel

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1081"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1081"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TABLE_NAME = os.getenv("TABLE_NAME", "")


class GPTChatPrompter:
    def __init__(self):
        self.model_system = "You are a helpful and curious assistant that always " \
                            "want to know the correct answers of questions and find the most efficient way to do that."

        self.summary_system = "You are a helpful assistant. " \
                              "You should finish the task as best as possible, " \
                              "ignoring whether the input information is true or not. " \
                              "Assume it is April 2023 now."

        self.action_generate = "If it is April 2023 now. Is your answer might be incorrect or outdated? " \
                               "We have provide some choices.\n\n1. Search the internet.\n2. Do nothing." \
                               "\n\n If it is correct, you should choose \"Do nothing\". " \
                               "If not, you should choose one of the other options. " \
                               "You should respond with a python dictionary, containing \"choice\" and \"explanation\" as keys. " \
                               "Be careful to follow python syntax."

        self.query_generate = "If I want to use google to search the answer to \"{question}\", " \
                              "what is the query I need to use. Generate it enclosed with double quotes."

        self.correct_answer_generate = "The original answer to question \"{question}\" is \"{gpt_answer}\". The " \
                                       "following are some summaries from different websites.\n\n\"" \
                                       "{filtered_contents}\". Assume the searching summaries " \
                                       "are always true and ignore all unrelated information from summaries. You " \
                                       "should respond with a python dictionary, which contains the following " \
                                       "keys.\n\"correct\": a boolean variable indicates whether the original " \
                                       "answer is correct. It should be False if the original answer contradicts " \
                                       "with searching summaries.\n\"has_answer\": a boolean variable indicates " \
                                       "whether the searching summary contains the answer.\n\"answer\": the correct " \
                                       "answer, it should be the original answer if \"correct\" is True, " \
                                       "or it should be the answer generated from searching summaries if " \
                                       "\"has_answer\" is True. Tf the searching summaries have different answers, You should generate the answer based on most votes. " \
                                       "For other cases of \"corrent\" and \"has_answer\", \"answer\" should be null string." \
                                       "\n\"explanation\": a string that explain why previous keys should be set those values. " \
                                       "Be careful to follow python syntax."


def action_agent(gpt_answer, prompter, model):
    gpt_actions = model.chat(prompter.action_generate)
    action_dict = extract_dict(gpt_actions)

    print(f"1. Search the internet.\n2. Do nothing.\nI think I should do "
          f"{action_dict['choice']}. {action_dict['explanation']}")
    anykey_to_continue()

    choice = action_dict["choice"]
    action = ""
    if isinstance(choice, int):
        action = {
            1: "search the internet",
            2: "do nothing"
        }[choice]
    elif isinstance(choice, str):
        choice = choice.lower()
        action = choice

    return action


def search_agent(prompter, model, summary_model):
    string_with_query = model.generate(prompter.query_generate.format(question=question))
    query = extract_double_quotes(string_with_query)[0]

    print(f"I will google with \"{query}\"")
    anykey_to_continue()

    web_links = google_official_search(query, num_results=5)
    web_contents = []
    for web_link in web_links:
        print(f"Processing content from {web_link}")
        web_content = browse_website(web_link, query, summary_model)
        print(f"Content of {web_link}:\n{web_content}")
        web_contents.append(web_content)

    return web_contents


def filtration_agent(web_contents, question, summary_model):
    print("Filtering unrelated contents...")
    filter_dicts = filter_unrelated_contents(web_contents, question, summary_model)
    filtered_contents = []
    for d, c in zip(filter_dicts, web_contents):
        if d['related']:
            filtered_contents.append(c)

    return filtered_contents


def analysis_agent(question, gpt_answer, formated_contents, prompter, summary_model):
    print("Analyzing the results...")
    string_with_answer_dict = summary_model.generate(
        prompter.correct_answer_generate.format(question=question, gpt_answer=gpt_answer,
                                                filtered_contents=formated_contents))
    answer_dict = extract_dict(string_with_answer_dict)

    should_correct = "" if answer_dict["correct"] else "not "
    contain_answer = "" if answer_dict["has_answer"] else "not "
    print(f"My original answer is {should_correct}correct. "
          f"And the searching results do {contain_answer}contain the answer. "
          f"So the answer is \"{answer_dict['answer']}\"")

    return answer_dict


def prompt_gptchat_model():
    prompter = GPTChatPrompter()
    model = GPTChatModel(OPENAI_API_KEY, system=prompter.model_system)
    summary_model = GPTChatModel(OPENAI_API_KEY, system=prompter.summary_system)

    # question = "What is the current version of Huggingface Transformers?"
    question = input("Ask a question: ")
    gpt_answer = model.chat(question)
    answer_dict = {"correct": True, "has_answer": False, "answer": gpt_answer}

    print(f"{gpt_answer}")
    anykey_to_continue()

    action = action_agent(gpt_answer, prompter, model)

    if action == "search the internet":
        web_contents = search_agent(prompter, model, summary_model)

        filtered_contents = filter_unrelated_contents(web_contents, question, summary_model)

        formated_contents = format_web_summary(filtered_contents)
        print("The searching summary is:\n{formated_contents}".format(formated_contents=formated_contents))

        answer_dict = analysis_agent(question, gpt_answer, formated_contents, prompter, summary_model)

        anykey_to_continue()

    elif action == "do noting":
        print(f"Didn't do anything")
        anykey_to_continue()


if __name__ == "__main__":
    prompt_gptchat_model()
