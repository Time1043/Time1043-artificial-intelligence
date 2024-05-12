# machine-learning

- 参考

  [numpy org](https://numpy.org/)、[pandas org](https://pandas.pydata.org/)、

  [matplotlib org](https://matplotlib.org/)、[seaborn org](https://seaborn.pydata.org/)、[echarts org](https://echarts.apache.org/zh/index.html)、

  [sklearn org](https://scikit-learn.org/)、scipy org、[statsmodels org](https://www.statsmodels.org/stable/index.html)

  [机器学习白板推导](https://www.bilibili.com/video/BV1aE411o7qd/)、[同学笔记](https://www.yuque.com/books/share/f4031f65-70c1-4909-ba01-c47c31398466?#)、
  
  [Jack-Cherish/Machine-Learning](https://github.com/Jack-Cherish/Machine-Learning)
  
  



- 环境

  ```bash
  conda create -n ml python==3.8
  conda activate ml
  
  pip install numpy pandas matplotlib 
  
  ```
  
  







# numpy















# pandas







# matplotlib









# seaborn













# Echarts













# sci-kit learn









# Machine Learning

- 机器学习的概念

  机器学习的分类：监督学习、无监督学习

- 机器学习的流程

  数据、划分训练集测试集

  训练集训练、损失函数、优化器、调参

  测试集测试、评估

- 机器学习的算法

  KNN、线性算法、决策树、神经网络、支持向量机、贝叶斯方法、集成学习

  聚类算法、主成分分析、概率图模型





- 机器学习的分类 (按学习方式)

  监督学习：人工标注的数据 (分类 回归)

  无监督学习：无需人工标注的数据 (聚类 异常点检测 可视化及降维 关联算法)

  半监督学习：监督 + 无监督

  

- 评价模型

  分类任务指标：accuracy、error、精确率、召回率、F1度量、ROC曲线

  回归任务指标：MSE、RMSE





- 学习

  想法、数学、代码

  



## 核心概念

- 总览 (需要基础)

  损失函数、梯度下降、决策边界、过拟合欠拟合、

  学习曲线、交叉验证、模型误差、正则化、LASSO和岭回归、模型泛化、

  评价指标 (混淆矩阵 精确率 召回率 ROC曲线)

  



## K-Nearest Neighbors

- 总览

  



## Linear Algorithm





## Decision Tree







## Neural Network (more)













## Support Vector Machine







## Bayes Method





## Ensemble Learning





## Clustering Algorithm





## Principal Component Analysis









## Probabilistic Graphical Model



- 总览

  “见微知著 睹始知终" (韩非)

  地位：贝叶斯学派、机器学习的集大成 统一各种模型

  理念：将 学习任务 转换为 计算变量的概率分布

  应用：不确定性推理、统计机器学习、语音识别、计算机视觉、自然语言处理

  

- 概率图模型

  概率模型为机器学习打开了一扇大门

  实际情况中各个变量间存在显示或隐式的相互依赖

  直接基于训练数据求解变量联合概率分布困难

  概率图模型 即用图来表示变量概率依赖关系

- 概率 + 图

  节点：随机变量

  边：变量间的条件概率分布

- 概率图分类

  directed graph：边有方向 因果关系 

  - Static Bayesian Network: `Naive Bayes`
  - Dynamic Bayesian Network (DBN): `Hidden Markov Model` (HMM), `Kalman Filter`
  
  undirected graph：无向边 相关关系 
  
  - Markov Random Field (MRF): `Boltzmann Machine`, `Conditional Random Field` (CRF)



- 概率图模型的求解

  最优化逼近

  Expectation-Maximization 参数估计

- 代码实现 (隐马尔可夫模型)



- 模型评价

  优缺点和适用条件





### 想法

- 概率图模型主要步骤

  表示 representation：将实际问题建模为图结构(求所有节点变量的联合概率分布 $p(X)=p(x_1,x_2,...x_n)$ )  

  推断 inference：简化去计算感兴趣图节点的后验概率分布(conditional probability distribution $p(X_A|X_B)$  marginal distribution $p(X_A)$  $p(X_B)$)  

  学习 learning：估计模型的参数(MLE MAP)





- Directed Graph (贝叶斯网络)

  节点：对应连续或离散随机变量

  有向边：连接父子节点、表示条件概率分布、不存在回路 (Directed Acyclic Graph)

  只认直系父母 其他祖先都简化

  统一：线性模型、决策树、神经网络 (本质都是有向图模型)

  后续：隐马尔可夫模型、卡尔曼滤波、因子分析、概率主成分分析、独立成分分析、混合高斯

  应用：最早的专家系统

- UnDirected Graph (马尔科夫随机场)

  节点：对应连续或离散随机变量

  无向边：两条有向边的组合、表示依赖关系(非父子)

  团clique：任意两节点都有边连接，则称节点子集为clique

  联合概率分布能基于clique分解
  
- interconversion

  Directed Graph -> Undirected Graph  - Moralization(道德化)

  Undirected Graph -> Directed Graph



- Markov 

  Markov Chain、Markov Property

  条件概率的独立假设
  
- Hidden Markov Model

  Hidden 

  Markov





- 推断方法

  精确推断：变量消去法、信念传播法 Belief Propagation

  近似推断：MCMC采样法、变分推断 Variational Inference (实用)





### 算法

- Expectation-Maximization (变分推断)

  隐变量参数的似然函数 $L(\theta)$

  对期望似然函数的最大化

- Step - loop (爬楼梯)

  E: q分布不变、最大化z

  M: z不变、q最大化寻优

- Evaluate

  反复迭代、可能陷入局部最优

  受初始值的影响





- Hidden Markov Model

  对序列数据建模

- Hidden Markov Model 基本假设

  观测独立性假设  $P(o_1,o_2,...,o_t|s_1,s_2,...,s_t)=\prod_{i=1}^{t}P(o_i|s_i)$  (任意时刻的观测只依赖于该时刻的马尔科夫链)

  齐次马尔可夫假设  $P(s_t|s_{t-1},s_{t-2},...,s_1)=P(s_t|s_{t-1})$  (t时刻的状态只与t-1时刻的状态有关)

- Markov Chain

  | 隐式链 (state) | 人称代词 | 动词 | 名词   |
  | -------------- | -------- | ---- | ------ |
  | 显式链         | 我       | 是   | 中国人 |

- Hidden Markov Model 三大要素

  状态初始概率

  状态转移矩阵

  发射概率矩阵

  

- 代码实现

  数据准备

  ```python
  import numpy
  
  # data preparation
  state = np.array(["认真复习", "简单复习", "没有复习"])
  grade = np.array(["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"])
  n_state, m_grade = len(state), len(grade)
  pi = np.ones(n_state) / n_state
  t = np.array([
      [0.4, 0.3, 0.3],
      [0.3, 0.4, 0.3],
      [0.3, 0.3, 0.4]
  ])
  e = np.zeros([3, 9])
  e[0, :9] = 1/9
  e[1, 3:9] = 1/6
  e[2, 5:9] = 1/4
  
  print(f"初始概率矩阵：\n{pi}\n")
  print(f"转移矩阵：\n{t}\n")
  print(f"发射矩阵：\n{e}\n")
  
  ```
  
  模型训练
  
  ```python
  # pip install hmmlearn
  from hmmlearn.hmm import CategoricalHMM
  
  hmm = CategorialHMM(n_state)
  hmm.startprob_ = pi
  hmm.transmat_ = t
  hmm.emissionprob_ = e
  hmm.n_feature = 9
  
  datas = np.array([0, 4, 2, 6, 1])
  datas = np.expand_dims(datas, axis=1)
  
  # Predict the hidden state sequence
  states = hmm.predict(datas)
  print(states)  
  
  # Predict the probability of observation series
  prob = hmm.score(datas)
  print(prob)  # ln
  print(np.exp(prob))  
  
  # Generate conditional constraint data
  datas, states = hmm.sample(10000)
  # check
  t_2 = np.zeros([3, 3])
  for i in range(3):
      current = np.where(states == i)[0]
      next_index = current + 1
      next_index = next_index[:-1]
      
      tmp = states[next_index]
      for j in range(3):
          t_2[i][j] = np.where(tmp == j)[0].shape[0] / np.shape(tmp)[0]
  print(t_2)
  # check
  e_2 = np.zeros([3, 9])
  for i in range(3):
      current = np.where(states == i)[0]
      next_index = current + 1
      next_index = next_index[:-1]
      
      tmp = datas[next_index]
      for j in range(3):
          e_2[i][j] = np.where(tmp == j)[0].shape[0] / np.shape(tmp)[0]
  print(e_2)
  
  ```
  
  
  
- 总结

  预测隐藏状态序列
  
  预测观测序列概率
  
  生成条件约束数据





### 评价

- HMM 简介

  提出：为了解决序列标注问题

  定位：理想的概率生成模型 (两个假设)

- HMM 优缺

  优点：大大简化条件概率计算 (时序数据建模)

  缺点：假设太强，大多数场景不适用



- PGM 简介

- PGM 优点

  思路清晰：建模表示 + 推断学习

  可解释性强，白盒子模型

- PGM 缺点

  难以确定节点间的拓扑关系

  推断和学习复杂，高维数据处理困难



- PGM 未来方向

  









## 项目实战

### 手搓神经网络

















# 机器学习白板推导









































