# AutoLearn-GPT: GPT learns to improve itself

| [EN](https://github.com/Reason-Wang/SelfLearnGPT/blob/main/README.md) | [中文](https://github.com/Reason-Wang/SelfLearnGPT/blob/main/docs/README-ZH.md) |

## 😕What is AutoLearn-GPT

This is an experimental project to explore whether ChatGPT model can learn by itself. Currently we equip the model with a memory brain to store everything that ChatGPT may not know. When using the model to finish tasks, it first recall (retrieve) relevant knowledge and use that as supplementary materials.

## 🧰How to use

+ We tested the project with python 3.8. Install all the required packages with `pip install -r requirements.txt`

+ Copy `.env.template` to `.env`
+ Get your *openai api key*, *pinecone api key*, *google api key* and *google search engine id* and set them in .env (If you don't know how to get these id and keys, you can refer to Auto-GPT documentation [here](https://significant-gravitas.github.io/Auto-GPT/configuration/search/)) 
+ Start running with `python main.py`

## 🎶Demo

https://user-images.githubusercontent.com/72866053/234460423-9946dfba-ccf6-4a79-813d-1341430b2168.mp4

## 📖How it works

1. Ask a question to the model, and the model tries to answer the question using information from its memory
2. If the answer might be incorrect, search the internet to collect information
3. The model filters unrelated information, and try to generate an answer from filtered information
4. The model compare the new answer with its original answer. if the original is correct, the model memorizes the new answer

![how_it_works](https://user-images.githubusercontent.com/72866053/234168105-97f9cdb1-78c4-4b25-b02b-009966782d57.png)
