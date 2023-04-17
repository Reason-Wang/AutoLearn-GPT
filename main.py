from commands.execute import execute_command
from model.vicuna.modeling import Vicuna

def main():
    model_name = 'eachadea/vicuna-13b-1.1'
    cache_dir = '../vicuna/cache'
    gpt_model = Vicuna(model_name, max_new_tokens=512, cache_dir=cache_dir)

    question = "What is the current version of HuggingFace Transformers?"
    gpt_answer = gpt_model.chat(question)

    # answer_correctness_prompt = f"One answered the question \"{question}\" with \"{gpt_answer}\". Is this the correct answer?"
    # answer_correctness_check = gpt_model.chat(answer_correctness_prompt)

    # do_action_prompt = "If you always want to know the correct answer. Based on your last response, select one action to do next:\n1. Search the internet for answer.\n2. Do nothing"

    #----
    # do_action_prompt = "Can you determine this is the correct answer? If you can, do nothing. If you can not, what should I do to find the correct answer? I can do one of the following actions.\n1. Search the internet for answer.\n2. Do nothing."
    # do_action = gpt_model.chat(do_action_prompt)
    #
    # do_action_check_prompt = f"The following is a suggestion: \n\n{do_action}\n\nDoes the suggestion mentioned any of the following actions to do\n1. Search the internet\n2. Do noting\n3. None of above."
    # do_action_check = gpt_model.generate(do_action_check_prompt)
    #---

    # command_name = get_command_from_output(do_action_check)
    results = execute_command("Search", question, gpt_model)


if __name__=="__main__":
    main()

