# hungyi-lee

- Reference

  [2015 Machine Learnging](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html)、

  [2024 Introduction to Generative Artificial Intelligence](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)

- Book

  [Neural Networks and Deep Learning (Neural Networks and Deep Learning)](http://neuralnetworksanddeeplearning.com/)

  [Deep Learning (Written by Yoshua Bengio, Ian J. Goodfellow and Aaron Courville)](http://www.iro.umontreal.ca/~bengioy/dlbook/)





# 2015 Machine Learnging

## Introduction

- Big Picture

  Machine Learning 

  Deep Learning 

  Structured Learning 

  



- Machine Learning 

  Program: 电脑只能做你命令他做的、你不会解的复杂任务他也无能为力

  - complex task: speech recognition, image recognition, Handwritten Recognition 

  Learning: **looking for a function**

- Framework

  - Training Data: function input, function output `{(x1,y1),(x2,y2),...}`

    Model: hypothesis function set `f1,f2,...`

    Training: pick the best function `f*`

  - Testing: best function `f*(x')=y'`

  



- Deep Learning (一种方法)

  Production line: 由许多简单函数串联的、能够完成复杂功能的函数

- Deep VS Shallow

  Shallow: hand-crafted (EX: MFCC)

  Deep: learned from data 

- Why Deep? Better! Why?

  Deep > simply (more parameters)

  Deep > Fat (parameters or trarning data)

  



- Structured Learning (一类问题)

  input domain: sequence, graph, tree...

  output domain: sequence, graph, tree...

- Example

  Retrieval: keyword -> a list of  web pages

  Translation: one kind of sequence -> another kind of sequence 

  Speech Recognition: one kind of sequence -> anothor kind of sequence 

  Speech Summarization: select the most informative segments to from a compact version 

  Object Detection: image -> object positions

  Pose Estimation: image -> pose

  



## Neural Network (Basic Ideas)

- Learning: looking for a function

  Example

  Framework

- Question

  What is the model? What is the function hypothesis set? 

  What is the "best" function? 

  How to pick the "best" function?

- Task

  Binary Classification

  Multi-class Classification: handwriting digit classification, image recongnition

  



- What is the model?

  $y = f(x) = \sigma(W^L...\sigma(W^2\sigma(W^1x + b^1) + b^2)...+b^L)$

  matrix function, derivation for compound function...

- Task: handwriting digit classification

  x: image -> [0,1,...] (16*16)

  y: [1,0,0,...] (10), [0,1,0,...] (10),... -> label 

- History

  A Layer of Neuron: Linear Transformation, Activation function 

  Limitation of Single Layer: simple logic gate problem, Hidden Neurons

  Neural Network: notation

  



- What is the "best" function? 

  best parameters!

- analyse

  function set: $y = f(x; \theta)$

  best function: $y = f^*(x; \theta^*)$; 

  Cost funtion: $C(\theta)$; $\theta^* = \mathop{\arg\min}\limits_{\theta} C(\theta)$

  



- How to pick the "best" function?

  Gradient Descent  

- 









# 2018 Generative Adversarial Network











# 2018 Deep Reinforcement Learning 













# 2019 Machine Learnging











# 2020 Deep Learning for Human Language Processing











# 2021 Machine Learnging









# 2022 Machine Learnging









# 2023 Machine Learnging











# 2024 Introduction to Generative Artificial Intelligence

## 背景概念

### 课程说明

- 机器对文章的分类

  示范demo

  嵌入应用

- 课程定位 (大学的课程期盼有长时效)

  日标受众：用ChatGPT不用学、生成式AI的原理及未来可能

  预修课程：不要求数学、不要求机器学习、不要求编程

  作业规划：体验生成式AI打造应用

  



### 生成式AI

- 生成式AI

  



### 大语言模型

- 大语言模型

  



## 今日的生成式AI厉害在哪里

- 今日的生成式人工智能厉害在哪里

  从 **工具** 变成 **工具人**





