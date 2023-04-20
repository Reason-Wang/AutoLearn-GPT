import openai


class GPTChatModel:
    def __init__(self, openai_api_key, system="You are a helpful assistant.", temperature=0.3):
        openai.api_key = openai_api_key
        self.model="gpt-3.5-turbo"
        self.system = system
        self.temperature = temperature
        self.conv = [
            {"role": "system", "content": system}
        ]

    def chat(self, user_input, temperature=None):
        self.conv.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.conv,
            temperature=self.temperature if temperature is None else temperature

        )
        self.conv.append({
            "role": "assistant",
            "content": response["choices"][0]["message"]["content"]
        })

        return response["choices"][0]["message"]["content"]

    def generate(self, user_input, temperature=None):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user_input}
            ],
            temperature=self.temperature if temperature is not None else temperature
        )

        return response["choices"][0]["message"]["content"]


