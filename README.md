## 😕What is SelfLearnGPT

This is an experimental project to explore whether GPT models can learn by themselves.

## 🧰How to use

+ Install all the required packages `pip install -r requirements.txt`

+ copy .env.template to .env
+ get your openai api key, pinecone api key, google api key and google search engine id and set them in .env
+ start running with `python main.py`

## 🎶Demo

https://user-images.githubusercontent.com/72866053/234065836-c725abb3-0d69-4ca4-9385-9e2a08f53c2a.mp4

## 📖How it works

1. Ask a question to the model, and the model tries to answer the question with its memory
2. If the answer might be incorrect, search the internet to collect information
3. The model filters unrelated information, and try to generate an answer from filtered information
4. The model compare the new answer with its original answer. if the original is correct, the model memorizes the new answer