

def extract_action_from_gpt_actions(gpt_actions):
    if "search the internet" in gpt_actions:
        return "search the internet"
