import logging
import os

from commands.browse import browse_website
from commands.search import google_official_search
from continuous_prompt.utils import extract_dict, extract_double_quotes, anykey_to_continue
from model.openai.modeling_chat import GPTChatModel

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1081"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1081"
openai_api_key = "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"


def prompt_gptchat_model():
    system_prompt = "You are a helpful and curious assistant that always want to know the correct answers of questions and find the most efficient way to do that."
    model = GPTChatModel(openai_api_key, system=system_prompt)

    summary_system_prompt = "You are a helpful assistant. You should finish the task as best as possible, ignoring whether the input information is true or not. Assume it is April 2023 now."
    summary_model = GPTChatModel(openai_api_key, system=system_prompt)

    # question = "What is the current version of Huggingface Transformers?"
    question = input("Ask a question: ")
    gpt_answer = model.chat(question)
    print(f"{gpt_answer}")
    anykey_to_continue()

    action_generate_prompt = f"If it is April 2023 now. Is your answer might be incorrect or outdated? We have provide some choices.\n\n1. Search the internet.\n2. Do nothing.\n\n If it is correct, you should choose \"Do nothing\". If not, you should choose one of the other options. You should respond with a python dictionary, containing \"choice\" and \"explaination\" as keys. Be careful to follow python syntax."
    gpt_actions = model.chat(action_generate_prompt)
    action_dict = extract_dict(gpt_actions)
    print(f"1. Search the internet.\n2. Do nothing.\nI think I should do {action_dict['choice']}. {action_dict['explanation']}")
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

    if action == "search the internet" or choice == "search the internet":
        query_generate_prompt = f"If I want to use google to search the answer to \"{question}\", what is the query I need to use. Generate it enclosed with double quotes."
        string_with_query = model.generate(query_generate_prompt)
        query = extract_double_quotes(string_with_query)[0]

        print(f"I will google with \"{query}\"")
        anykey_to_continue()

        web_links = google_official_search(query, num_results=2)
        web_link = web_links[0]
        web_content = browse_website(web_link, query, summary_model)

        print(f"Content of {web_link}:\n{web_content}")
        anykey_to_continue()
    # Option 1: Use original model
    # correct_answer_to_question_prompt = f"Your original answer to question \"{question}\" is \"{gpt_answer}\". The searching summary is \"{web_content}\". If the searching summary is always true, is your original answer correct? You should respond with a python dict, which contains the following keys.\n\"correct\": a boolean variable indicates whether the original answer is correct.\"has_answer\": a boolean variable indicates whether the searching summary contains the answer.\n\"answer\": the correct answer, should be your original answer if your original answer is true, or the answer from searching summary if it contains the correct answer, otherwise should be null string."
    # string_with_answer_dict = model.chat(correct_answer_to_question_prompt)
    # answer_dict = extract_double_quotes(string_with_answer_dict)

    # Option 2: Use a new model, seems to be better
    correct_answer_to_question_prompt = f"The original answer to question \"{question}\" is \"{gpt_answer}\". The web searching summary is \"{web_content}\". If the searching summary is always true, is the original answer correct? You should respond with a python dict, which contains the following keys.\n\"correct\": a boolean variable indicates whether the original answer is correct.\"has_answer\": a boolean variable indicates whether the searching summary contains the answer.\n\"answer\": the correct answer, it should be your original answer if \"correct\" is True, or it should be the answer from searching summary if \"has_answer\" is True, otherwise should be null string. Be careful to follow python syntax."
    string_with_answer_dict = model.generate(correct_answer_to_question_prompt)
    answer_dict = extract_dict(string_with_answer_dict)

    should_correct = "" if answer_dict["correct"] else "not"
    print(f"My original answer is {should_correct} correct. So the answer is \"{answer_dict['answer']}\"")
    anykey_to_continue()
    # Seems not need anymore
    if False:
        should_learn_prompt = f"For the question \"{question}\", your original answer is \"{gpt_answer}\". And your answer after searching the internet is \"{answer}\"\nIs you original answer correct?"
        should_learn = model.chat(should_learn_prompt)


if __name__ == "__main__":
    prompt_gptchat_model()