# AutoLearn-GPT: GPT learns to improve itself

[![Stargazers repo roster for @Reason-Wang/AutoLearn-GPT](https://reporoster.com/stars/Reason-Wang/AutoLearn-GPT)](https://github.com/Reason-Wang/AutoLearn-GPT/stargazers)

[[EN](https://github.com/Reason-Wang/SelfLearnGPT/blob/main/README.md) | [简中](https://github.com/Reason-Wang/SelfLearnGPT/blob/main/docs/README-ZH.md)]

## 😕什么是AutoLearn-GPT

这个一个探究GPT模型是否可以通过自身学习的实验性项目。当前我们给ChatGPT模型配备了一个记忆大脑来存储一切ChatGPT可能不知道的内容。当使用ChatGPT完成任务时，首先会回忆(检索)相关的知识作为补充材料。

## 🧰如何使用

+ 我们在python 3.8版本测试了项目。通过`pip install -r requirements.txt`安装所需的包。

+ 复制`.env.template`文件，并命名为`.env`
+ 获取OpenAI的api key，pinecone的api key，google api key和google search engine id。(如果你不知道如何获取这些api key或id，可以参考Auto-GPT的[文档](https://significant-gravitas.github.io/Auto-GPT/configuration/search/)。) 
+ 通过`python main.py`运行项目（如果位于中国大陆地区，需要使用代理才可正常运行）

## 🎶演示

https://user-images.githubusercontent.com/72866053/234460647-ff63242a-539e-4035-891a-aa659b2b57e0.mp4

## 📖原理

1. 问模型一个问题，模型通过回忆已有的记忆并回答问题
2. 如果回答可能是不正确的，通过网络搜索相关信息
3. 模型过滤掉不相干的信息，通过已过滤的信息回答问题
4. 模型比较新的答案和原本的回答，如果原本的答案不正确，模型将记住现有的答案

![how_it_works_zh](https://user-images.githubusercontent.com/72866053/234229415-272352da-0df4-40b9-842f-2979c9e36a1d.png)
