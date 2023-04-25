# SelfLearnGPT

------

[EN](https://github.com/Reason-Wang/SelfLearnGPT/blob/main/README.md) | [中文](https://github.com/Reason-Wang/SelfLearnGPT/blob/main/docs/README-ZH.md)

## 😕什么是SelfLearnGPT

这个一个探究GPT模型是否可以通过自身学习的实验性项目

## 🧰如何使用

+ 我们在python 3.8版本测试了项目。通过`pip install -r requirements.txt`安装所需的包。

+ 复制`.env.template`文件，并命名为`.env`
+ 获取OpenAI的api key，pinecone的api key，google api key和google search engine id。(如果你不知道如何获取这些api key或id，可以参考Auto-GPT的[文档](https://significant-gravitas.github.io/Auto-GPT/configuration/search/)。) 
+ 通过`python main.py`运行项目

## 🎶演示

https://user-images.githubusercontent.com/72866053/234065836-c725abb3-0d69-4ca4-9385-9e2a08f53c2a.mp4

## 📖原理

1. 问模型一个问题，模型通过回忆已有的记忆并回答问题
2. 如果回答可能是不正确的，通过网络搜索相关信息
3. 模型过滤掉不相干的信息，通过已过滤的信息回答问题
4. 模型比较新的答案和原本的回答，如果原本的答案不正确，模型将记住现有的答案
