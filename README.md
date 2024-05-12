# artificial-intelligence
- 介绍

  这是学习阶段中有关人工智能的学习。



- 学习阶段

  课程入门：速通

  自主学习：论文阅读

- 学习知识

  机器学习：

  深度学习：
  
  强化学习：





- Reference github

  [ultralytics yolov8](https://github.com/ultralytics/ultralytics),  [paddleOCR](https://github.com/PaddlePaddle/PaddleOCR),  [chatgpt-on-wechat](https://github.com/zhayujie/chatgpt-on-wechat), [al_yolo cheat](https://github.com/EthanH3514/AL_Yolo), [rapidOCR-json](https://github.com/hiroi-sora/RapidOCR-json)

- Reference book

  [CS自学指南](https://csdiy.wiki/)

  [CS142: Web Applications (Spring 2023)](https://web.stanford.edu/class/cs142/)、

  李航 统计学习方法、

  [李宏毅2024 生成式AI导论](https://www.youtube.com/watch?v=Q9cNkUPXUB8&t=145s)、

  [李宏毅2023 生成式AI](https://www.youtube.com/watch?v=yiY4nPOzJEg&list=PLJV_el3uVTsOePyfmkfivYZ7Rqr2nMk3W)、[ML2023spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)、

  李宏毅2022、

  [李宏毅2021](https://www.youtube.com/watch?v=Ye018rCVvOo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J)、

  李宏毅2019、

  李宏毅2018、

  [李沐 斯坦福21秋季 实用机器学习中文版](https://www.bilibili.com/video/BV13U4y1N7Uo/)、[课程主页](https://c.d2l.ai/stanford-cs329p/)、

  [李沐 动手学深度学习 PyTorch版](https://www.bilibili.com/video/BV1KA411N7Px/)、

  [西湖大学 赵世钰 强化学习](https://www.bilibili.com/video/BV1r3411Q7Rr)、

  [浙大 胡浩基 强化学习](https://www.bilibili.com/video/BV1FM411w7kT)、

  [一个月吃透机器学习](https://www.bilibili.com/video/BV14j421U7oW/)、

  阿里

- 论文参考

  axiv、CVPR、[ieee](https://www.ieee.org/)、[springer](https://www.springer.com/us)、[scihub](https://sci-hub.hkvisa.net/)、谷歌学术、谷粉学术

  [CVPR2023放榜之际汇总回顾CVPR2022与遥感相关的论文](https://www.sohu.com/a/648684161_121123740) 

  [CNN改进](https://zhuanlan.zhihu.com/p/656424068)





# Machine Learning

## 感知机 (神经网络)

- 介绍

  找到一个能把训练数据按照类别分隔开的 **超平面**

  ![](res/Snipaste_2024-04-02_09-49-34.png)

  参数：w、b

  求得参数的方法：最小化损失、梯度下降



- 手搓代码

  1





## k-Nearest Neighbor

- 介绍

  物以类聚人以群分

  通过在特征空间中找到与新样本 **最近** 的训练样本来预测其类别



- 手搓代码

  ```python
  # -*- coding: utf-8 -*-
  # @Time    : 2024/4/2 8:19
  # @Author  : yingzhu
  # @FileName: knn.py
  
  import numpy as np
  import pandas as pd
  from time import time
  
  
  def load_data(file_path):
      """
      加载数据集
      :param file_path:
      :return:
      """
      print(f"[start loading data] {file_path}...")
      features, labels = [], []
      data = pd.read_csv(file_path)
      data_list = data.values.tolist()
  
      for item in data_list:
          labels.append(item[0])
          features.append(item[1:])
      print(f"[features shape] {len(features)}, \n[labels shape] {len(labels)}")
      print(f"[sample features] {features[0]}, \n[sample labels] {labels[0]}")
      return features, labels  # [[#(28*28)],[#(28*28)],[#(28*28)],...], [1,2,3,...]
  
  
  def calculate_distance(x1, x2):
      """
      计算两个特征向量之间的距离 欧式距离
      :param x1: np.array
      :param x2: np.array
      :return:
      """
      return np.sqrt(np.sum(np.square(x1 - x2)))  # 相减 平方 求和 开方
  
  
  def get_closest(x, features, labels, k=5):
      """
      获取k个最近邻的标签和距离
      :param x:
      :param features: [[#(28*28)],[#(28*28)],[#(28*28)],...]
      :param labels: [1,2,3,...]
      :param k:
      :return:
      """
      dist_list = np.zeros(len(features))
      for i, feat in enumerate(features):
          dist = calculate_distance(np.array(x), np.array(feat))
          dist_list[i] = dist
  
      top_index = np.argsort(np.array(dist_list))[:k]  # 索引排序 [4,3,2,1] -> [3,2,1,0]
      candidates = [labels[i] for i in top_index]
      res = max(candidates, key=candidates.count)  # 出现次数最多的标签
      return res
  
  
  def model_test(train_features, train_labels, test_features, test_labels, k=5):
      """
      模型测试
      :param train_features:
      :param train_labels:
      :param test_features:
      :param test_labels:
      :param k:
      :return: accuracy
      """
      print("[start testing]...")
      error_count = 0
      for i in range(len(test_features)):
          x = test_features[i]
          y = get_closest(x, train_features, train_labels, k)
          if y != test_labels[i]: error_count += 1
      return 1 - (error_count / len(test_features))
  
  
  if __name__ == '__main__':
      print("------------------ start ------------------")
  
      start_time = time()
      file_path_train = "../data/mnist_train.csv"
      file_path_test = "../data/mnist_test.csv"
      features_train, labels_train = load_data(file_path_train)
      features_test, labels_test = load_data(file_path_test)
      print(f"load data cost time: {time() - start_time:.2f}s")
  
      acc = model_test(features_train, labels_train, features_test, labels_test, k=30)
      print(f"testing cost time: {time() - start_time:.2f}s")
      print(f"accuracy: {acc:.4f}")
  
      print("------------------ end ------------------")
      
  ```

  1





## 朴素贝叶斯





## 逻辑回归





## PageRank









## 【案例】垃圾邮件过滤







## 【案例】法律判决预测









## 【案例】新闻摘要



















# Deep Learning

## CNN

### LeNet





### AlexNet





### VGGNet







### GoogleNet







### ResNet





### DenseNet













## RNN

### RNN (1986)



### LSTM (1997)





## Attention

### Transformer (2017)





### Bert





### GPT



### ViT



### winTransformer







### Mamba

- 论文：[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

- 回顾总结

  |             | Training Phase | Testing Phase | Additional Issue       |
  | ----------- | -------------- | ------------- | ---------------------- |
  | RNN         | Slow           | Fast          | Rapid Forgetting       |
  | LSTM        | Slow           | Fast          | Forgetting             |
  | Transformer | Fast           | Slow          | Ram & Time: O(n^2)     |
  | Mamba       | Fast           | Fast          | NO  (Ram & Time: O(n)) |

  





## Diffusion















# YOLO

















# Large Language Model

## 大模型的科普

### 概念



- 时髦的产品

  AIGC：AI生成内容GeneratedContent (文本 代码 图片 音频 视频)  

  gpt、githubCopilot、midjourney

  https://github.com/features/copilot

  https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F



- 概念BigPicture

  ![Snipaste_2023-10-21_17-13-57](res/Snipaste_2023-10-21_17-13-57.png)

- 机器学习下的各种学习

  机器学习：区分与显式编程

  监督学习：

  无监督学习：

  强化学习：行为、环境、状态

  深度学习：神经网络

  ![Snipaste_2023-10-21_17-20-45](res/Snipaste_2023-10-21_17-20-45.png)

- 深度学习的应用：AIGC、LLM

  生成式AI：用神经网络识别现有内容和结构，学习生成新的内容 (文本 代码 图片 音频 视频)  

  大语言模型：自然语言处理任务

  - 大：模型的参数量很大、海量的训练数据
  - 根据输入提示和前面生成过的词，通过概率计算生成下一个token，以生成文本序列

  (不是所有的大语言模型都适合文本生成，GoogleBERT理解上下文的能力很强) -> 谷歌搜索 情感分析 文本分类

  (生成图像的扩散模型不是大语言模型)

  ![Snipaste_2023-10-21_17-22-14](res/Snipaste_2023-10-21_17-22-14.png)



### 原理：Attention机制

- 任务：生成、分类、总结、改写

- 需要：

  大量文本进行无监督学习

  模型巨大的参数量

  



- 语言模型架构的演变：

  RNN：RecurrentNeuralNetwork

  - 按顺序逐字处理，每一步的输出取决于先前的隐藏状态和当前的输入
  - 要等待上一个步骤完成后才能进行当前计算，无法并性计算，训练效率低 -> ?改串联为并联
  - 容易忘，不擅长处理长序列，难以捕获长距离的语义关系 -> RNN改良：LSTM

  Transformer 

  - (GPT：GenerativePre-trained-Transformer)
  - 学习输入序列里所有词的相关性和上下文，不受到短时记忆的影响

- 下一个出现概率最大的词是什么？GPT、搜索引擎

  Q：模型如何得到各个词出现的概率？





- Transformer的两个核心概念：

  注意力机制Attention(理解每个词的意义)

  位置编码(捕获词在句子的位置) —— 独立计算

- Transformer的两个核心组件：

  编码器Encoder

  解码器Decoder

- Transformer的训练步骤：

  词嵌入：输入文本token化，每个token词转化成整数tokenID，传入嵌入层使得每个token都用向量表示 (维度信息：词的语义、词间关系)

  位置编码：对词向量进行位置编码=词向量+位置向量 (理解每个词的意义、捕获词在句中的位置 -> 理解不同词间的顺序关系)

  编码器：【理解】把输入转换成更抽象的表现形式，用自注意机制捕捉关键特征 (保留词汇信息和顺序关系 捕捉了关键特征)

  - 多头自注意力层

    Q：自注意力机制：计算每对词间的相关性，来决定注意力权重 (权重会学习调整)

    A：解码器输出是关注了这个词本身的信息、还包含上下文中的相关信息

    Q：自注意机制的计算？

    Q：多头？每个头都有自己的注意力权重，用来关注文本里不同的特征和方面 (动词 修饰词 情感 命名实体) - 并行计算

  - 前馈神经网络：对自注意力模块的输出进行处理，增强模型的表达能力

  - 多个堆叠的编码器：每个编码器内部结构一样、但不共享权重，使得模型能更深入理解数据、处理更复杂的语言内容

  解码器：【写作】生成一个个词！

  - 输入：解码器输出的抽象token、一个特殊值start(表示输出序列的开头) —— 把之前已生成的文本也作为输入，保证输出的连贯性和上下文相关性

  - (对已生成的文本进行词嵌入和位置编码)

  - 带掩码的多头自注意力层：

    A：只针对已生成的输出序列

    Q：带掩码：只关注该词和之前的词、后面的词要遮住 —— 确保解码器生成文本时遵循正确的时间顺序 预测下一个词只使用前面的作为上下文条件

  - 多头自注意力层：

    A：针对编码器输出的抽象token

    A：捕捉编码器的输出和解码器即将生成的输出之间的对应关系，将原始输入序列信息融合到输出序列的生成过程

  - 前馈神经网络：通过额外的计算增强模型的表达能力

  - 多个堆叠的解码器：增加模型性能，有助于处理复杂的输入输出关系

  最后的Linear和softmax层：把解码器输出的表示转化为词汇表的概率分布 (下一个被生成token的概率)

- Transformer的本质和弊端：

  编码器：理解和表示输入序列

  解码器：生成输出序列  

  解码器本质上是在猜下一个最可能的输出，但输出是否符合客观事实，模型无从得知 (普通的算术题都不会)

  ![Snipaste_2023-12-21_10-59-54](res/Snipaste_2023-12-21_10-59-54.png)

- Transformer的变种：

  仅编码器/自编码模型：BERT，掩码语言建模、情感分析 (理解语言的任务)

  仅解码器/自回归模型：GPT，文本生成任务

  编码器解码器/序列到序列模型：T5、BART，翻译、总结 (把一个序列转化成另一个序列)





- GPT的训练步骤：

  无监督学习预训练：大量文本，得到基座模型 (文本生成 但不擅长对话)

  - 自己找出数据中的结构和模式，自行学习人类语言的语法语义表达模式
  - 基于条件去预测、由预测和期望得到损失、由损失更新权重参数 -> 迭代
  - 最耗时、费力、烧钱！！！

  监督微调：人类撰写的高质量对话数据，得到【微调后的基座模型SFT】 (续写文本 更好的对话能力)

  - 喂给模型问题+人类中意的回答 (监督学习SupervisedFine-Tuning)

  - 改变模型的内部参数，让模型更加适应特定任务
  - 需要的数据更少、训练时间更短、成本更低！！！

  强化学习准备：问题+多个对应回答+人类标注，训练【奖励模型】 (能对回答进行评分预测)

  强化学习训练：【微调后的基座模型】【奖励模型】 

  - 智能体、环境；行动、奖励、状态

  - 模型在环境中采取行动、获得结果反馈、从反馈中学习
  - 奖励打分：helpful有用性、honest真实性、harmless无害性











## 大模型的应用



- 2024年AI工具

  最炫酷的AI工具 LUMA AI： https://lumalabs.ai/

  看起来最有用的AI工具 Gamma：https://gamma.app/

  最好用的AI搜索引擎 Phind：https://www.phind.com/

  最有潜力的AI vall-e-x：https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-x/

  最好用的AI学习工具 Albus：https://albus.org/zh/

  使用最多的AI工具 Raycast AI：https://www.raycast.com/

  程序员最爱的AI工具 Warp：https://www.warp.dev/

  最有用的AI工具 Notion：https://www.notion.so/product/ai

  最好听的AI工具 Sono AI：https://app.suno.ai/

  最好用的AI浏览器 Arc：https://arc.net/

  最受期待的AI Sora：https://openai.com/sora

  智能体：https://www.gnomic.cn/?laowang





### API调用







![Snipaste_2024-01-21_21-41-43](res/Snipaste_2024-01-21_21-41-43.png)



- 使用AI模型：网页服务、代码交互 (云端服务)

- 网页服务的困境：

  网页内容的抓取存储应用 

  提示词提示模板

  给AI的指示参数不灵活 (回复长度、创造性、频率惩罚)

  大量数据的批处理

  AI模型的集成和定制化 (自动回复邮件、生成报告)



- AI模型在服务器上：API调用

  http请求响应：自行构建http请求、格式是api规定好的

  封装python库：调用相关函数方法

  - [openai-chatGPT-APIdocs](https://platform.openai.com/docs/api-reference)、[百度-文心-APIdosc](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2)
  - 

- API密钥：创建获取、追踪计费

  创建密钥：[openai-chatGPT-APIkey](https://platform.openai.com/api-keys)

  隐藏明文：设置本机环境变量 `OPENAI_API_KEY`

- API计费：提示+回答的token

  计算花费：[基于token数量](https://platform.openai.com/tokenizer)、[各类模型计费](https://openai.com/pricing)

  限制token：代码设置、网页的ContextWindow





- **max_tokens**：回答所消耗token的最大值、到达到后直接截断

  **temperatue**：AI回答的随机性创造性、

  - 

  

  另外

  **best_of**: 用于生成多个答案并从中选择最佳答案。这个参数决定了生成答案的数量。更高的值意味着从更多的答案中选择，但也会增加计算成本。

  **n**: 决定生成的回答数量。例如，如果设置为 5，模型会生成 5 个不同的回答供选择。

  **logprobs**: 这个参数可以用来请求模型提供关于其预测的更多信息。例如，设置为 10，模型将返回每个生成的 token 的前 10 个最可能的 token 的概率。

  **echo**: 如果设置为 true，模型会在输出中包含输入的内容。

  **stop**: 可以指定一个或多个序列，当模型生成这些序列时会停止。这对于定义回答的结尾非常有用，比如在生成文本的特定部分或段落后停止。

  **user**: 用于传递关于调用者的信息，这可以帮助模型更好地理解和回应特定用户的需求。

  要删

  1. **prompt**: 输入给模型的文本，模型根据这个文本生成回应。
  2. **max_tokens**: 指定模型输出的最大 token 数量。
  3. **temperature**: 控制生成文本的随机性，范围从 0 到 1。
  4. **top_p**: 控制生成的文本多样性，范围从 0 到 1。
  5. **n**: 指定要生成的回答数量。
  6. **logprobs**: 请求模型提供每个 token 的概率分布信息。
  7. **echo**: 如果设置为 true，输出将包含输入的文本。
  8. **stop**: 定义生成文本时的停止符或序列。
  9. **presence_penalty**: 调整模型生成新颖内容的倾向。
  10. **frequency_penalty**: 调整模型避免重复内容的倾向。
  11. **best_of**: 生成多个回答并从中选择最佳答案。
  12. **user**: 提供有关用户的信息，帮助模型更好地定制回应。
  13. **stream**: 决定是否以流的方式返回输出，适用于长文本生成。
  14. **return_prompt**: 决定是否在响应中包含输入的提示文本。
  15. **expand**: 用于在生成文本时扩展或调整特定的部分。
  
- 代码实现

  ```python
  from openai import OpenAI
  import tiktoken
  import erniebot
  
  
  def req_openai():
      # 创建openai实例
      client = OpenAI(base_url='https://api.aigc369.com/v1')  # 读取环境变量的api_key
  
      # 发送请求：model、messages
      completion = client.chat.completions.create(
          model='gpt-3.5-turbo',
          response_format={'type': 'json_object'},
          messages=[
              {'role': 'system', 'content': '你是一个CS专业的老教授，你十分喜欢教学和琢磨内容'},  # system系统消息  背景角色
              # {'role': 'user', 'content': '进阶sql有哪些学习的内容'},
              {'role': 'user', 'content': '给我以BigPicture的形式，总结一下动态规划需要掌握的知识'},  # user表示用户发送的  指示提示
              # {'role': 'assistant', 'content': '我是chatGPT，由openai开发的一款大语言模型'},  # assistant表示AI发送的  回答
          ]
      )
  
      # 得到响应
      print(completion.choices)  # ChatCompletion类实例
      print(completion.choices[0].message)
  
      """
      进阶sql有哪些学习的内容
      ChatCompletionMessage(content='进阶SQL学习的内容包括：\n\n1. 子查询：深入学习如何在查询中嵌套使用子查询，以实现更复杂的查询逻辑。\n\n2. 联结（JOIN）查询：掌握多个表之间的关系，学习如何使用不同类型的JOIN操作（如内连接、左连接、右连接、全连接）来获取联结结果。\n\n3. 窗口函数：了解并应用窗口函数，可以在查询结果的基础上进行排序、分组、计数、累计等操作。\n\n4. 索引和性能优化：学习如何设计并使用索引来提高查询效率，并了解一些性能优化的技巧，如使用合适的查询语句、避免全表扫描等。\n\n5. 视图：了解视图的概念和用途，并学习如何创建和使用视图来简化复杂的查询，提高查询的可复用性。\n\n6. 存储过程和函数：学习如何创建和使用存储过程和函数，可以在数据库中定义可重用的代码块，进一步提高开发效率。\n\n7. 事务管理：了解事务的概念和特性，并学习如何使用事务控制语句（如BEGIN、COMMIT、ROLLBACK）来确保数据库操作的一致性和完整性。\n\n8. 数据库管理：学习如何进行数据库备份和恢复、性能监控和调优、安全控制等数据库管理相关的操作。\n\n9. 高级查询技巧：学习一些高级查询技巧，如使用CASE语句、使用WITH语句进行递归查询、使用GROUP BY和HAVING子句进行分组和过滤等。\n\n10. 数据库设计原则：了解数据库设计的原则和规范，包括数据模型设计、表关系建立、范式化等，以及如何进行数据库重构和优化。\n\n以上是进阶SQL学习的一些内容，具体的学习内容和难度可以根据个人需求和实际情况进行调整和深入学习。', role='assistant', function_call=None, tool_calls=None)
      """
  
      """
      进阶sql有哪些学习的内容
      ChatCompletionMessage(content='进阶SQL学习的内容包括：\n\n1. 复杂查询：学习使用JOIN操作来连接多个表，使用子查询和联合查询等技术来实现复杂查询需求；\n2. 数据库设计和规范化：学习如何设计和规范化数据库，包括确定关系和实体，选择适当的数据类型，设计合适的主键和外键等；\n3. 索引和性能优化：学习使用索引来提高查询性能，并学习如何分析和优化查询的执行计划；\n4. 存储过程和触发器：学习创建和使用存储过程和触发器，以实现复杂的业务逻辑；\n5. 安全性和权限管理：学习如何设置数据库用户和角色，以及如何控制他们的权限和访问级别；\n6. 数据备份和恢复：学习如何备份和恢复数据库，以及如何进行数据迁移和复制；\n7. 数据库分区：学习如何使用分区技术来管理大型数据库，提高查询性能和维护效率；\n8. 高级数据类型和函数：学习如何使用高级数据类型（如数组、JSON、XML等）和函数来处理和查询复杂数据；\n9. OLAP和数据仓库：学习使用OLAP（联机分析处理）和数据仓库技术来处理大量的历史数据和复杂的分析需求；\n10. SQL优化和调试：学习如何识别和解决SQL查询中的性能问题和错误。\n\n以上是进阶SQL学习的一些内容，根据个人需求和实际情况，还可以选择学习其他特定的数据库管理系统（如MySQL、Oracle、SQL Server等）的高级功能和特性。', role='assistant', function_call=None, tool_calls=None)
      """
  
      """
      panda和spark、flink等对数据处理有哪些共通的和区别
      Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='panda、Spark和Flink都是用于数据处理的工具，它们有一些共通的和区别。以下是它们的一些共通和区别点：\n\n共通点：\n1. 处理大规模数据：Pandas、Spark和Flink都能够处理大规模的数据，它们提供了分布式计算和并行处理的能力，能够有效地处理大量的数据。\n\n2. 数据处理功能：这些工具都提供了丰富的数据处理功能，如数据清洗、转换、过滤、聚合等操作。\n\n3. 数据分析：Pandas、Spark和Flink都支持数据分析任务，它们能够进行统计计算、机器学习、图计算等高级数据分析任务。\n\n区别点：\n1. 编程模型：Pandas是Python的一个数据处理库，使用的是基于单机的DataFrame编程模型。而Spark和Flink是分布式计算框架，使用的是基于集群的数据流编程模型。\n\n2. 运行环境：Pandas是在单机上运行的，适合处理中小规模的数据。而Spark和Flink是运行在分布式集群上的，能够处理大规模数据，并且具有良好的可扩展性。\n\n3. 实时处理：Flink具有流式处理的特性，能够实时处理数据流，支持事件时间和处理时间的语义。而Spark在早期主要关注离线批处理，不过现在也已经提供了流处理的功能。\n\n4. 数据结构：Pandas是基于表格型数据结构的，非常适合进行类似于关系型数据库的操作。而Spark和Flink使用的是分布式数据集（RDD和DataSet/DataStream），它们更适合处理大规模的、非结构化的数据。\n\n总的来说，Pandas适合处理中小规模的数据，Spark适合大规模的离线批处理和流处理，而Flink则更加适合实时流处理。具体选择哪个工具需要根据具体的应用场景和需求来决定。', role='assistant', function_call=None, tool_calls=None))
      ChatCompletionMessage(content='panda、Spark和Flink都是用于数据处理的工具，它们有一些共通的和区别。以下是它们的一些共通和区别点：\n\n共通点：\n1. 处理大规模数据：Pandas、Spark和Flink都能够处理大规模的数据，它们提供了分布式计算和并行处理的能力，能够有效地处理大量的数据。\n\n2. 数据处理功能：这些工具都提供了丰富的数据处理功能，如数据清洗、转换、过滤、聚合等操作。\n\n3. 数据分析：Pandas、Spark和Flink都支持数据分析任务，它们能够进行统计计算、机器学习、图计算等高级数据分析任务。\n\n区别点：\n1. 编程模型：Pandas是Python的一个数据处理库，使用的是基于单机的DataFrame编程模型。而Spark和Flink是分布式计算框架，使用的是基于集群的数据流编程模型。\n\n2. 运行环境：Pandas是在单机上运行的，适合处理中小规模的数据。而Spark和Flink是运行在分布式集群上的，能够处理大规模数据，并且具有良好的可扩展性。\n\n3. 实时处理：Flink具有流式处理的特性，能够实时处理数据流，支持事件时间和处理时间的语义。而Spark在早期主要关注离线批处理，不过现在也已经提供了流处理的功能。\n\n4. 数据结构：Pandas是基于表格型数据结构的，非常适合进行类似于关系型数据库的操作。而Spark和Flink使用的是分布式数据集（RDD和DataSet/DataStream），它们更适合处理大规模的、非结构化的数据。\n\n总的来说，Pandas适合处理中小规模的数据，Spark适合大规模的离线批处理和流处理，而Flink则更加适合实时流处理。具体选择哪个工具需要根据具体的应用场景和需求来决定。', role='assistant', function_call=None, tool_calls=None)
      """
  
      """
      给我以BigPicture的形式，总结一下动态规划需要掌握的知识
      [Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='当涉及到动态规划（Dynamic Programming）时，以下是你需要掌握的核心知识：\n\n1. 递归与迭代关系：理解递归和迭代之间的联系以及它们在解决问题中的应用。\n\n2. 最优子结构：理解最优子结构的概念，即问题具有可分解成互不依赖的子问题并可以通过组合子问题的最优解来得到整体问题的最优解。\n\n3. 重叠子问题性质：了解动态规划问题中常常存在的重叠子问题，并懂得如何通过记忆化或表格来避免重复计算。\n\n4. 状态转移方程：掌握如何定义问题的状态、状态之间的关系以及状态转移方程的构建，这是动态规划问题的核心。\n\n5. 问题拆解与求解顺序：学会将复杂的问题拆解为多个简单的子问题，并确定求解子问题的顺序，其中的关键在于处理子问题时所需的其他子问题的结果是否已经求解完毕。\n\n6. 基本解法：了解基本的动态规划解法方法，如自顶向下的记忆化搜索（Top-Down Memoization）和自底向上的迭代求解（Bottom-Up Iteration）。\n\n7. 空间优化：学会通过状态压缩、滚动数组等技巧来减少动态规划解法中所需的额外空间。\n\n8. 常见问题类型：熟悉常见的动态规划问题类型，如背包问题、最长递增子序列、编辑距离等，并掌握相应的解决思路和算法。\n\n9. 实战经验：通过实际问题的练习和实战，不断积累动态规划的应用经验，并学会优化算法以提高效率。\n\n总结起来，掌握动态规划需要了解其核心概念和解题思路，熟悉相关的算法技巧，并通过实践不断提升自己的解题能力。', role='assistant', function_call=None, tool_calls=None))]
      ChatCompletionMessage(content='当涉及到动态规划（Dynamic Programming）时，以下是你需要掌握的核心知识：\n\n1. 递归与迭代关系：理解递归和迭代之间的联系以及它们在解决问题中的应用。\n\n2. 最优子结构：理解最优子结构的概念，即问题具有可分解成互不依赖的子问题并可以通过组合子问题的最优解来得到整体问题的最优解。\n\n3. 重叠子问题性质：了解动态规划问题中常常存在的重叠子问题，并懂得如何通过记忆化或表格来避免重复计算。\n\n4. 状态转移方程：掌握如何定义问题的状态、状态之间的关系以及状态转移方程的构建，这是动态规划问题的核心。\n\n5. 问题拆解与求解顺序：学会将复杂的问题拆解为多个简单的子问题，并确定求解子问题的顺序，其中的关键在于处理子问题时所需的其他子问题的结果是否已经求解完毕。\n\n6. 基本解法：了解基本的动态规划解法方法，如自顶向下的记忆化搜索（Top-Down Memoization）和自底向上的迭代求解（Bottom-Up Iteration）。\n\n7. 空间优化：学会通过状态压缩、滚动数组等技巧来减少动态规划解法中所需的额外空间。\n\n8. 常见问题类型：熟悉常见的动态规划问题类型，如背包问题、最长递增子序列、编辑距离等，并掌握相应的解决思路和算法。\n\n9. 实战经验：通过实际问题的练习和实战，不断积累动态规划的应用经验，并学会优化算法以提高效率。\n\n总结起来，掌握动态规划需要了解其核心概念和解题思路，熟悉相关的算法技巧，并通过实践不断提升自己的解题能力。', role='assistant', function_call=None, tool_calls=None)
      """
  
  
  def cal_token():
      """ openai官方的token分词，不需要消耗token """
      encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')  # 传入模型名 返回对应编码器
      res = encoding.encode('周公恐惧流言日，王莽谦恭未篡时。')
      print(res)  # tokenId列表  [41642, 35417, 21555, 238, 33565, 100, 89753, 78244, 9080, 3922, 29207, 233,
      print(len(res))  # 24token
  
  
  def req_wenxin_erniebot():
      resp = erniebot.ChatCompletion.create(
          model='ernie-3.5',
          messages=[
              {'role': 'user', 'content': '你觉得chatGPT厉害，还是文心一言厉害'}
          ]
      )
      print(resp.get_result())
  
  
  if __name__ == '__main__':
      req_openai()
      # cal_token()
  
  ```
  
  





### Prompt Engineering





![Snipaste_2024-01-21_23-52-26](res/Snipaste_2024-01-21_23-52-26.png)





- 






- 思维链：步子小点不容易扯着、把注意力集中在当前思考步骤上、减少上下文的过多干扰

  ![Snipaste_2024-01-22_14-21-37](res/Snipaste_2024-01-22_14-21-37.png)

  不用小样本提示，分步骤思考

  ![Snipaste_2024-01-22_14-26-19](res/Snipaste_2024-01-22_14-26-19.png)





### 应用开发实战

- AI模型本身无法记忆历史对话、不会阅读外部文档、不擅长数学运算不懂如何上网

  用代码武装AI模型：对话记忆、外部知识库、外部工具

  核心模型：国内国外模型、云端本地模型、开源闭源模型



- LangChain

  AI模型的输入与输出：模型、提示模板、小样本提示模板、输出解析器

  AI模型的记忆：对话缓冲记忆、对话缓冲窗口记忆、对话摘要记忆、对话摘要缓冲记忆、对话令牌缓冲记忆

  AI模型读取外部文件：索引增强生成、

  AI模型调用外部工具：



- Assistant API

  Streamlit网站搭建

  AI应用部署



### 【应用】视频脚本生成器





### 【应用】小红书AI写作助手







### 【应用】克隆chatGPT







### 【应用】CSV数据分析智能工具







### 【应用】智能PDF问答工具





### 国内大模型

- 百度千帆

  [百度千帆平台](https://console.bce.baidu.com/qianfan/)、[百度千帆API文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2)

  



- zhipu AI安装环境

  ```
  
  ```

- 快速入门

  



- tongyi





## 扩展旁支

### assistant API





### Streamlit 网站搭建





### AI 应用部署











## LangChain

- 定位

  大语言模型开发框架：应用程序方便快捷地与LLM对接、脱离LLM复杂的API细节、专注

- 参考

  [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)、[LangChain支持的LLM](https://python.langchain.com/docs/integrations/llms/)
  
  [langchain中通义千问的使用](https://python.langchain.com/docs/integrations/llms/tongyi)、[通义千问key申请方法](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)、
  
  [langchain中星火大模型的使用](https://python.langchain.com/docs/integrations/llms/sparkllm)、[星火大模型的api接口申请地址](https://xinghuo.xfyun.cn/sparkapi)、[星火API文档](https://www.xfyun.cn/doc/spark/Web.html)
  
  [langchain中的文心一言](https://python.langchain.com/docs/integrations/llms/baidu_qianfan_endpoint)、[文心一言的key申请（收费）](https://cloud.baidu.com/)



- [LangChain Architecture](https://python.langchain.com/docs/get_started/introduction)

  官方支持语言：python、js

  旁支：LangSmith (监控调试)、LangServe (快速部署web服务)、Templates (内置模板快速创建场景)

  核心：LangChain (应用程序接合)、LangChain-Community (集成第三方模块)、LangChain-Core (表达式语言、批处理、异步……)

  ![](res/Snipaste_2024-04-02_21-55-01.png)

- [安装环境](https://python.langchain.com/docs/get_started/installation)

  ```
  conda create -n llm python==3.9
  conda activate llm
  
  pip install langchain
  pip install langchain-cli 
  
  pip install langchain-openai
  pip install --upgrade --quiet  dashscope -i https://mirrors.aliyun.com/pypi/simple
  pip install dashscope -i https://mirrors.aliyun.com/pypi/simple
  pip install qianfan -i https://mirrors.aliyun.com/pypi/simple
  
  ```

- 快速入门

  [openai](https://python.langchain.com/docs/integrations/llms/openai)
  
  ```python
  import os
  
  from langchain_openai import ChatOpenAI
  from langchain_core.prompts import ChatPromptTemplate
  
  os.environ["http_proxy"] = "http://127.0.0.1:7890"
  os.environ["https_proxy"] = "http://127.0.0.1:7890"
  
  llm = ChatOpenAI()
  llm.invoke("how are you")  # AIMessage 对象
  
  ```
  
  [sparkLLM](https://python.langchain.com/docs/integrations/llms/sparkllm)
  
  ```python
  import os
  
  import yaml
  from langchain_community.llms import SparkLLM
  
  yaml_file = "../key/key.yaml"
  with open(yaml_file, 'r') as file:
      data = yaml.safe_load(file)
  spark_info = data.get("spark", {})
  os.environ["IFLYTEK_SPARK_APP_ID"] = spark_info.get('APPID')
  os.environ["IFLYTEK_SPARK_API_SECRET"] = spark_info.get('APISecret')
  os.environ["IFLYTEK_SPARK_API_KEY"] = spark_info.get('APIKey')
  
  llm_spark = SparkLLM()
  res = llm_spark.invoke("嘉靖有一句话：云在青天水在瓶，你觉得是什么意思?")
  print(res)
  ```
  
  [Tongyi](https://python.langchain.com/docs/integrations/llms/tongyi)
  
  [Zhipu](https://python.langchain.com/docs/integrations/chat/zhipuai)
  
  [Baidu Qianfan](https://python.langchain.com/docs/integrations/llms/baidu_qianfan_endpoint)
  
  
  
  1











# Reinforcement Learning

- 概念

  强化学习：基于经验的学习

  环境 Environment、智能体/代理 Agent、观测 Observation、状态 State、奖励 Reward、动作/决策 Action/Decision

- probabilistic graphical model



























# Paper

- arxiv论文下载

  ```python
  # -*- coding: utf-8 -*-
  # @Time    : 2024/4/1 22:00
  # @Author  : yingzhu
  # @FileName: download_paper.py
  
  import os
  import re
  import time
  from urllib.parse import urlparse
  
  import requests
  from bs4 import BeautifulSoup
  
  
  def get_paper_title(url):
      response = requests.get(url)
      soup = BeautifulSoup(response.text, 'html.parser')
      title = soup.find('h1', class_='title mathjax').get_text()
      title = title.replace('Title:', '').strip()  # Removing 'Title:' prefix
  
      # Cleaning and formatting the title for use in filenames
      title = re.sub('[^a-zA-Z0-9 \n\.]', '', title)  # Remove invalid characters
      title = re.sub(' +', ' ', title).strip()  # Remove extra spaces
      title = title[:50]  # Limit title length if necessary
      return title
  
  
  def download_pdfs(url_list, destination_folder, delay=5):
      if not os.path.exists(destination_folder):
          os.makedirs(destination_folder)
          print(f"Created directory: {destination_folder}")
  
      downloaded_files = []
      headers = {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
      }
  
      for url in url_list:
          try:
              paper_title = get_paper_title(url)
              parsed_url = urlparse(url)
              file_name = parsed_url.path.split('/')[-1] + f"_{paper_title}.pdf"
              file_path = os.path.join(destination_folder, file_name)
              pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
  
              print(f"Downloading {pdf_url}...")
  
              response = requests.get(pdf_url, headers=headers)
              if response.status_code == 200:
                  with open(file_path, 'wb') as f:
                      f.write(response.content)
                  downloaded_files.append(file_path)
                  print(f"Downloaded and saved to {file_path}")
              else:
                  print(f"Failed to download PDF from {url}. Status Code: {response.status_code}")
  
          except Exception as e:
              print(f"An error occurred while downloading {url}: {e}")
  
          time.sleep(delay)
  
      return downloaded_files
  
  
  def read_urls_from_file(file_path):
      with open(file_path, 'r') as file:
          urls = file.readlines()
      return [url.strip() for url in urls]
  
  
  if __name__ == '__main__':
      print("------------------ start ------------------")
  
      file_path = r"D:\code2\python-code\artificial-intelligence\paper.txt"
      url_list = read_urls_from_file(file_path)
  
      destination_folder = r"crawl\papers"
      downloaded_files = download_pdfs(url_list, destination_folder)
      print("Downloaded files:", downloaded_files)
  
      print("------------------ end ------------------")
  
      """
      https://arxiv.org/abs/1904.11486
      https://arxiv.org/abs/1811.11168
      """
  
  ```

  1













































































































































































