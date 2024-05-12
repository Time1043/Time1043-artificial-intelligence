# mu-li

- Reference

  [paper-reading](https://github.com/mli/paper-reading)、[paper-reading-note](https://github.com/Tramac/paper-reading-note)、

  [Dive into Deep Learning D2L.ai](https://github.com/d2l-ai/d2l-zh)、[D2L.ai book](http://zh.d2l.ai/)

  [UC Merced phd](https://bryanyzhu.github.io/)





# D2L.ai

## 预备知识

### 数据操作

- 数据操作

  获取数据、读取数据

- n维数组 (万物皆数)

  0-d scalar: a class  `shape(1)`

  1-d vctor: eigenvector  `shape(n)`

  2-d matrix: a sample / eigenmatrix  `shape(m,n)`

  3-d tensor: rgb picture  `shape(channel,height,width)`

  4-d tensor: a batch of rgb pictures  `shape(batch,channel,height,width)`

  5-d tensor: a batch of viedo  `shape(batch,time,channel,height,width)`

- 创建数组

  形状：`shape(3,4)`

  每个元素的数据类型：float32

  每个元素的值：全0，或随机数(正态分布 均匀分布)

- 访问元素

  一个元素：`matix[1,2]`

  一行：`matix[1,:]`

  一列：`matix[:,1]`

  子区域：`matix[1:3,1:]`

  子区域：`matix[::3,::2]`

- 代码实现

  创建

  ```python
  import numpy as np
  import torch
  
  x = torch.arange(12)  # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
  print(x.shape)  # torch.Size([12])
  print(x.numel())  # 12
  
  # reshape
  X = x.reshape(3, 4)  # tensor([[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]])
  
  # ones and zeros
  tensor_ones = torch.ones(3, 5, 4)
  tensor_zeros = torch.zeros(3, 5, 4)
  
  # from list or ndarray
  tensor_from_list = torch.tensor([[[1, 12, 3], [4, 5, 6]]])
  tensor_from_ndarray = torch.from_numpy(np.array([[1, 12, 3], [4, 5, 6]]))
  print(tensor_from_list.shape)  # torch.Size([1, 2, 3])
  print(tensor_from_ndarray.shape)  # torch.Size([2, 3])
  
  ```

  运算 (对元素运算 连结 逻辑运算 维度不同广播机制)

  ```python
  import torch
  
  # + - * / ** to elements
  x = torch.tensor([1.0, 2, 4, 8])  # float
  y = torch.tensor([2, 2, 2, 2])
  print(x + y, x - y, x * y, x / y, x ** y)
  # more calculations to elements (example exp)
  print(torch.exp(x))
  
  # concatenate rows or columns
  X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
  Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
  print(torch.cat((X, Y), dim=0), "\n", torch.cat((X, Y), dim=1))  # shape(6, 4)  shape(3, 8)
  
  # build tensor with logical operations
  print(X == Y)  # shape(3, 4)  dtypes(bool)
  
  # sum the elements of tensor
  print(X.sum(), X.sum(dim=0), X.sum(dim=1))  # tensor(66.)  tensor([12., 15., 18., 21.])  tensor([ 6., 22., 38.])
  print(X.sum(axis=0, keepdim=True))  # shape(1, 4)  broadcasting mechanism (dim!)
  
  # broadcasting mechanism
  a = torch.arange(3).reshape(3, 1)
  b = torch.arange(2).reshape(1, 2)
  print(a, "\n", b)
  print(a + b)  # shape(3, 2)
  
  ```

  元素的访问 写入

  ```python
  import torch
  
  X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
  Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
  
  # reading elements
  print(X[-1], "\n", X[1:3])  # dim 0
  
  # writing elements
  X[:, 1] = 10
  print(X)  # dim 1
  
  ```

  内存分配问题

  ```python
  import torch
  
  X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
  Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
  
  # New memory address
  before = id(Y)  # pointer
  Y = Y + X
  print(before, id(Y), id(Y) == before)  # False
  
  # Reserved memory address
  Z = torch.zeros_like(Y)
  print("id(Z):", id(Z))
  Z[:] = Y + X
  print("id(Z):", id(Z))
  
  # Save memory overhead
  before = id(X)
  X += Y  # X[:] = X + Y
  print(before, id(X), id(X) == before)  # True
  
  ```

  数据转换

  ```python
  import torch
  
  X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
  Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
  
  # tensor -> ndarray
  A = X.numpy()
  B = torch.tensor(A)
  print(type(A), type(B))
  
  # tensor.shape(1) -> python scalar
  a = torch.tensor([3.5])
  print(a, a.item(), float(a), int(a))
  
  ```

  



### 数据预处理

- 创建人工数据集 存储在csv

  ```python
  import os
  
  import pandas as pd
  import torch
  
  # crate fake data
  os.makedirs(os.path.join("..", "data"), exist_ok=True)
  data_file = os.path.join("..", "data", "house_tiny.csv")
  with open(data_file, "w") as f:
      f.write("NumRooms,Alley,Price\n")  # column names
      f.write("NA,Pave,127500\n")  # row: sample data
      f.write("2,Oswin,106000\n")
      f.write("4,NA,178100\n")
      f.write("NA,NA,140000\n")
  
  # read data from csv
  data = pd.read_csv(data_file)
  print(data)
  
  # split data into input and outputs
  inputs, outputs = data.iloc[:, :-1], data.iloc[:, -1]
  print(inputs)
  # process missing values(int): interpolate, delete
  inputs = inputs.fillna(inputs.mean(numeric_only=True))
  print(inputs)
  # process missing values(str): add a class (one-hot encoding)
  inputs = pd.get_dummies(inputs, dummy_na=True)
  print(inputs)
  
  # dataframe(number) -> tensor
  for column in ['Alley_Oswin', 'Alley_Pave', 'Alley_nan']:
      inputs[column] = inputs[column].astype(bool).astype(int)
  X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
  print(X, "\n", y)
  
  # remove data directory
  os.remove(data_file)
  os.rmdir(os.path.join("..", "data"))
  
  ```

  



- Question 

  ```python
  import torch
  
  a = torch.arange(12)
  b = a.reshape(3, 4)
  b[:] = 2
  print(a)  # tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
  
  ```

  



### 基础数学

- Linrear Algebra

  scalar, vector, matrix, tensor

  Distance(Euclid, Manhattan, Chebyshev, Minkowski), Norm(Frobenius)

  Matrix Multiplication, Linear Transformations (Distortion of space)

  Symmetric matrix, Positive definite matrix, Orthogonal matrix, Permutation matrix

  eigenvalue, eigenvector

- axis

  shape(2, 5, 4)

  axis: 0, 1, 2  - 0是高维

  sum(axis=0), sum.shape(5, 4)  - 坍缩/降维

  sum(axis=1), sum.shape(2, 4)  

  sum(axis=[1, 2]), sum.shape(2)  

  

- Matrix derivation (Calculus)

  The derivative is the slope of the tangent line

  Scalar derivatives, vector derivatives, matrix derivatives!!

  gradient：梯度是和等高线正交的、梯度指向的是值变化最大的方向！！！

- vector derivatives (知道怎么算 不用自己算)

  y标量关于X列向量的导数 -> 行向量

  Y向量关于x列向量的导数 -> 列向量

  Y列向量关于X列向量的导数 -> 矩阵

  ![Snipaste_2024-04-29_20-55-44](res/Snipaste_2024-04-29_20-55-44.png)

  



### 自动微分

- 链式法则 (Layer)

  ![Snipaste_2024-04-29_21-12-19](res/Snipaste_2024-04-29_21-12-19.png)

- 求导方式 (计算一个函数在指定值上的导数)

  符号求导

  数值求导(很小h)

  自动求导(计算图)

- 自动求导的两种模式 (复杂度估计)

  构造计算图

  `正向积累`：执行图，存储中间结果

  `反向传递`：从相反方向执行图 (去除不需要的直)



- 编程

  ```python
  import torch
  
  # y = 2 * X^T * X
  x = torch.arange(4.0)  # tensor([0., 1., 2., 3.])
  x.requires_grad_(True)  # X = torch.arange(4.0, requires_grad=True)  save gradient
  print(x.grad)  # None
  
  # calculate
  y = 2 * torch.dot(x, x)
  print(y)  # tensor(28., grad_fn=<MulBackward0>) - autograd
  
  # backward
  y.backward()
  print(x.grad)  # tensor([ 0.,  4.,  8., 12.])
  
  # check
  print(x.grad == 4 * x)  # tensor([True, True, True, True])
  
  # default pytorch accumulates gradients, so we need to zero them manually
  x.grad.zero_()
  y = x.sum()
  y.backward()
  print(x.grad)  # tensor([1., 1., 1., 1.])
  
  ```

  非标量变量的反向传播???

  ```python
  # 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
  # 本例只想求偏导数的和，所以传递一个1的梯度是合适的
  x.grad.zero_()
  y = x * x
  # 等价于y.backward(torch.ones(len(x)))
  y.sum().backward()
  x.grad
  ```

  分离计算??? (把网络中参数固定住)

  ```python
  x.grad.zero_()
  y = x * x
  u = y.detach()
  z = u * x
  
  z.sum().backward()
  print(x.grad == u)
  
  x.grad.zero_()
  y.sum().backward()
  print(x.grad == 2 * x)
  ```

  Python控制流的梯度计算???

  ```python
  def f(a):
      b = a * 2
      while b.norm() < 1000:
          b = b * 2
      if b.sum() > 0:
          c = b
      else:
          c = 100 * b
      return c
  
  a = torch.randn(size=(), requires_grad=True)
  d = f(a)
  d.backward()
  ```

  



### 查阅文档

- 查阅文档

  为了知道模块中可以调用哪些函数和类

  ```python
  import torch
  
  print(dir(torch.distributions))
  ```

  有关如何使用给定函数或类的更具体说明

  ```python
  help(torch.ones)
  ```

  



## 线性神经网络

### 线性回归

- 房价预测 (简化模型)

  假设1：影响因素为卧室个数、卫生间个数、居住面积

  假设2：成交价是三个因素的加权求和 (w1 w2 w3 b)

  本质：单层的神经网络

- 衡量预估质量

  平方损失 (求导方便)

- 训练数据 (学习参数)

  损失最小 (凸函数)



- 优化方法

  梯度下降 mini-batch 



- 代码实现 (不用框架)

  ```python
  import random
  import torch
  from d2l import torch as d2l
  
  
  # step 0: 生成数据集
  def synthetic_data(w, b, num_examples):  # @save
      """生成y=Xw+b+噪声"""
      X = torch.normal(0, 1, (num_examples, len(w)))
      y = torch.matmul(X, w) + b
      y += torch.normal(0, 0.01, y.shape)
      return X, y.reshape((-1, 1))
  
  
  true_w = torch.tensor([2, -3.4])
  true_b = 4.2
  features, labels = synthetic_data(true_w, true_b, 1000)
  print('features:', features[0], '\nlabel:', labels[0])
  
  d2l.set_figsize()
  d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
  d2l.plt.show()
  
  
  # step 1: 读取数据集
  def data_iter(batch_size, features, labels):
      num_examples = len(features)
      indices = list(range(num_examples))
      # 这些样本是随机读取的，没有特定的顺序
      random.shuffle(indices)
      for i in range(0, num_examples, batch_size):
          batch_indices = torch.tensor(
              indices[i: min(i + batch_size, num_examples)])
          yield features[batch_indices], labels[batch_indices]
  
  
  batch_size = 10
  for X, y in data_iter(batch_size, features, labels):
      print(X, '\n', y)
      break
  
  # step 2: 初始化模型参数 定义模型 定义损失函数 定义优化算法
  w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
  b = torch.zeros(1, requires_grad=True)
  
  
  def linreg(X, w, b):  # @save
      """线性回归模型"""
      return torch.matmul(X, w) + b
  
  
  def squared_loss(y_hat, y):  # @save
      """均方损失"""
      return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
  
  
  def sgd(params, lr, batch_size):  # @save
      """小批量随机梯度下降"""
      with torch.no_grad():
          for param in params:
              param -= lr * param.grad / batch_size
              param.grad.zero_()
  
  
  # step 3: 训练模型
  lr = 0.03
  num_epochs = 3
  net = linreg
  loss = squared_loss
  
  for epoch in range(num_epochs):
      for X, y in data_iter(batch_size, features, labels):
          l = loss(net(X, w, b), y)  # X和y的小批量损失
          # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
          # 并以此计算关于[w,b]的梯度
          l.sum().backward()
          sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
      with torch.no_grad():
          train_l = loss(net(features, w, b), labels)
          print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
  
  print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
  print(f'b的估计误差: {true_b - b}')
  
  ```

  







## 多层感知机













## 深度学习计算









## 卷积神经网络

















## 现代卷积神经网络

- 现代卷积神经网络

  AlexNet 深度卷积神经网络

  VGG 使用重复元素的网络

  NiN 网络中的网络

  GoogLeNet 含并行连结的网络

  ResNet 残差网络

  DenseNet 稠密连接网络



### AlexNet 













## 物体检测 Object Detection

### 物体检测和数据集

- Task

  图片分类 images classification 

  物体检测 Object Detection (应用更多 标注成本高)
  
- 物体检测的应用

  无人驾驶、无人售后、机器人

  



- [边缘框 bounding box](http://zh.d2l.ai/chapter_computer-vision/bounding-box.html)

  用4个数字定义 `(左上x, 左上y, 右下x, 右下y)` 或 `(左上x, 左上y, 宽, 高)` 或 `(中间x, 中间y, 宽, 高)`

- 目标检测数据集

  每row表示一共物体 `(图片文件名, 物体类别, 边缘框)`

  [COCO](https://cocodataset.org/) 80物体 330K图片 1.5M物体

- 边缘框实现

  ```python
  %matplotlib inline
  import torch
  from d2l import torch as d2l
  
  d2l.set_figsize()
  img = d2l.plt.imread('../img/catdog.jpg')
  d2l.plt.imshow(img);
  ```

  多种定义方式相互转换

  ```python
  #@save
  def box_corner_to_center(boxes):
      """从（左上，右下）转换到（中间，宽度，高度）"""
      x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
      cx = (x1 + x2) / 2
      cy = (y1 + y2) / 2
      w = x2 - x1
      h = y2 - y1
      boxes = torch.stack((cx, cy, w, h), axis=-1)
      return boxes
  
  #@save
  def box_center_to_corner(boxes):
      """从（中间，宽度，高度）转换到（左上，右下）"""
      cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
      x1 = cx - 0.5 * w
      y1 = cy - 0.5 * h
      x2 = cx + 0.5 * w
      y2 = cy + 0.5 * h
      boxes = torch.stack((x1, y1, x2, y2), axis=-1)
      return boxes
  
  
  ```
  
  简单验证
  
  ```python
  # bbox是边界框的英文缩写
  dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
  
  boxes = torch.tensor((dog_bbox, cat_bbox))
  box_center_to_corner(box_corner_to_center(boxes)) == boxes
  ```
  
  画出边缘框
  
  ```python
  #@save
  def bbox_to_rect(bbox, color):
      # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
      # ((左上x,左上y),宽,高)
      return d2l.plt.Rectangle(
          xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
          fill=False, edgecolor=color, linewidth=2)
          
  fig = d2l.plt.imshow(img)
  fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
  fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
  ```
  
  



- [小数据集示范](http://zh.d2l.ai/chapter_computer-vision/object-detection-dataset.html)

  ```python
  %matplotlib inline
  import os
  import pandas as pd
  import torch
  import torchvision
  from d2l import torch as d2l
  
  #@save
  d2l.DATA_HUB['banana-detection'] = (
      d2l.DATA_URL + 'banana-detection.zip',
      '5de26c8fce5ccdea9f91267273464dc968d20d72')
  ```

  读取数据集

  ```python
  #@save
  def read_data_bananas(is_train=True):
      """读取香蕉检测数据集中的图像和标签"""
      data_dir = d2l.download_extract('banana-detection')
      csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                               else 'bananas_val', 'label.csv')
      csv_data = pd.read_csv(csv_fname)
      csv_data = csv_data.set_index('img_name')
      images, targets = [], []
      for img_name, target in csv_data.iterrows():
          images.append(torchvision.io.read_image(
              os.path.join(data_dir, 'bananas_train' if is_train else
                           'bananas_val', 'images', f'{img_name}')))
          # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
          # 其中所有图像都具有相同的香蕉类（索引为0）
          targets.append(list(target))
      return images, torch.tensor(targets).unsqueeze(1) / 256
  ```

  创建自定义数据集

  ```python
  #@save
  class BananasDataset(torch.utils.data.Dataset):
      """一个用于加载香蕉检测数据集的自定义数据集"""
      def __init__(self, is_train):
          self.features, self.labels = read_data_bananas(is_train)
          print('read ' + str(len(self.features)) + (f' training examples' if
                is_train else f' validation examples'))
  
      def __getitem__(self, idx):
          return (self.features[idx].float(), self.labels[idx])
  
      def __len__(self):
          return len(self.features)
  ```

  数据加载器

  ```python
  #@save
  def load_data_bananas(batch_size):
      """加载香蕉检测数据集"""
      train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                               batch_size, shuffle=True)
      val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                             batch_size)
      return train_iter, val_iter
  ```

  测试

  ```python
  batch_size, edge_size = 32, 256
  train_iter, _ = load_data_bananas(batch_size)
  batch = next(iter(train_iter))
  batch[0].shape, batch[1].shape
  ```

  展示具体的

  ```python
  imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
  axes = d2l.show_images(imgs, 2, 5, scale=2)
  for ax, label in zip(axes, batch[1][0:10]):
      d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
  ```

  



### 物体检测算法







## 循环神经网络









## 现代循环神经网络















## 注意力机制











## 优化算法

















## 计算性能











## 计算机视觉

















## 自然语言处理：预训练

















## 自然语言处理：应用





























