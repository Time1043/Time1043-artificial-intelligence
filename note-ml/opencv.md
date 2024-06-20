# opencv

- Reference

  [OpenCV Course - Full Tutorial with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o&t=135s), 



- Introduction

  > ⭐️ Code ⭐️
  > 🔗Github link: https://github.com/jasmcaus/opencv-course
  >
  > 🔗The Caer Vision library: https://github.com/jasmcaus/caer
  >
  > ⭐️ Course Contents ⭐️
  >
  > ⌨️ (0:00:00) Introduction
  >
  > ⌨️ (0:01:07) Installing OpenCV and Caer
  >
  > Section #1 - Basics
  >
  > ⌨️ (0:04:12) Reading Images & Video
  >
  > ⌨️ (0:12:57) Resizing and Rescaling Frames
  >
  > ⌨️ (0:20:21) Drawing Shapes & Putting Text
  >
  > ⌨️ (0:31:55) 5 Essential Functions in OpenCV
  >
  > ⌨️ (0:44:13) Image Transformations
  >
  > ⌨️ (0:57:06) Contour Detection
  >
  > Section #2 - Advanced
  >
  > ⌨️ (1:12:53) Color Spaces
  >
  > ⌨️ (1:23:10) Color Channels
  >
  > ⌨️ (1:31:03) Blurring
  >
  > ⌨️ (1:44:27) BITWISE operations
  >
  > ⌨️ (1:53:06) Masking
  >
  > ⌨️ (2:01:43) Histogram Computation
  >
  > ⌨️ (2:15:22) Thresholding/Binarizing Images
  >
  > ⌨️ (2:26:27) Edge Detection
  >
  > Section #3 - Faces:
  >
  > ⌨️ (2:35:25) Face Detection with Haar Cascades
  >
  > ⌨️ (2:49:05) Face Recognition with OpenCV's built-in recognizer
  >
  > Section #4 - Capstone
  >
  > ⌨️ (3:11:57) Deep Computer Vision: The Simpsons

  



- What you will learn

- Basics

  Reading images and video

  Image transformations

  Drawing shapes

- Advanced

  Color spaces

  BITWISE operations

  Masking 

  Histogram computation

  Edge detection

- Faces

  Face detection

  Face recognition

  Deep computer vision





- Module

  core: 最核心的数据结构

  highgui: 视频与图像的读取 显示 存储

  imgproc: 图像处理的基础方法

  features2d: 图像特征及特征匹配





- Prepare the environment

  ```bash
  conda create -n ocv python==3.8.5
  conda activate ocv
  
  pip install opencv-contrib-python
  pip install caer
  
  
  touch .gitignore
  mkdir section01-basics section02-advanced section03-faces section04-capstone
  
  cd section01-basics/
  touch read.py
  
  cd D:\code2\python-code\artificial-intelligence\opencv-tutorial\section01-basics
  python .\read.py
  
  ```
  
  









# opencv (bilibili)

- 安装环境

  ```
  pip install opencv-python
  
  ```

  



- 图片的读取 展示 保存

  ```python
  import cv2
  
  print(cv2.__version__)
  
  # 读取图片
  img = cv2.imread('data/170927_063834858_Camera_5.jpg', cv2.IMREAD_GRAYSCALE)
  print(type(img))  # <class 'numpy.ndarray'>
  print(img.shape)  # (1710, 3384)
  
  # 显示图片
  cv2.imshow('image', img)
  # k = cv2.waitKey(0)  # 等待显示
  # print(k)  # ASCII 码值
  
  # 保存图片
  if cv2.waitKey(0) == ord('s'):
      cv2.destroyAllWindows()
      cv2.imwrite('data/image_copy.jpg', img)   
      
  ```

  



- 边缘检测 Canny

  梯度 -> 亮度变化速度 -> 边缘

  各个方向的梯度 (四个方向)

  上阈值 下阈值 (强边缘, 弱边缘, 非边缘) 

  与强边缘相联的弱边缘是边缘 否则不是边缘

  ```python
  import cv2
  
  img = cv2.imread('data/image_copy.jpg', cv2.IMREAD_GRAYSCALE)
  
  # cv2.imshow('image', img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  edge_img = cv2.Canny(img, 230, 550)
  cv2.imshow('edge_image', edge_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  cv2.imwrite('data/edge_image.jpg', edge_img)
  ```

  调试

  ```python
  import cv2
  
  cv2.namedWindow('edge_detection')
  cv2.createTrackbar('minThreshold', 'edge_detection', 50,1000, lambda x: x)
  cv2.createTrackbar('maxThreshold', 'edge_detection', 100,1000, lambda x: x)
  
  img = cv2.imread('data/image_copy.jpg', cv2.IMREAD_GRAYSCALE)
  
  while True:
      minThreshold = cv2.getTrackbarPos('minThreshold', 'edge_detection')
      maxThreshold = cv2.getTrackbarPos('maxThreshold', 'edge_detection')
  
      edges = cv2.Canny(img, minThreshold, maxThreshold)
  
      cv2.imshow('edge_detection', edges)
      cv2.waitKey(10)
  ```

  



- 剔除无关信息 roi_mask

  region of interest 感兴趣的区域

  数组切片 布尔运算

  ```python
  import cv2
  import numpy as np
  
  edge_img = cv2.imread('data/edge_image.jpg', cv2.IMREAD_GRAYSCALE)
  # cv2.imshow('edge_image', edge_img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  # 掩码
  mask = np.zeros_like(edge_img)
  mask = cv2.fillPoly(mask, np.array([[[5,1400],[2668,1697],[1650,744],[1570,718]]]), color=255)
  cv2.imshow('mask', mask)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  # 布尔运算
  masked_edge_img = cv2.bitwise_and(edge_img, mask)
  cv2.imshow('masked_edge_img', masked_edge_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imwrite('data/masked_edge_image.jpg', masked_edge_img)
  
  ```

  



- 霍夫变换

  在图像中寻找圆或直线 

  从直角坐标系映射到极坐标系 通过极坐标的性质寻找直线

  ```python
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt
  
  
  # 霍夫变换
  edge_img = cv2.imread('data/masked_edge_image.jpg', cv2.IMREAD_GRAYSCALE)
  lines = cv2.HoughLinesP(edge_img, 1, np.pi/180, 15, minLineLength=40, maxLineGap=20)
  print(len(lines))  # 216条直线实际都表示的是同一条直线
  
  # 用matplotlib画出直线
  for line in lines:
      x1, y1, x2, y2 = line[0]
      plt.plot([x1, x2], [y1, y2], 'r-')
  plt.imshow(edge_img, cmap='gray')
  plt.show()
  
  # 用斜率来过滤 合并 (实际只有2条直线)
  def calculate_slope(line):
      x1, y1, x2, y2 = line[0]
      return (y2 - y1) / (x2 - x1)
  
  # 分组：按照斜率分成两组车道线
  left_lines = [line for line in lines if calculate_slope(line) > 0]
  right_lines = [line for line in lines if calculate_slope(line) < 0]
  print(len(left_lines), len(right_lines))
  
  # 去噪：离群值过滤 计算两组的平均斜率 若出现与平均斜率相差更大 则舍弃
  def reject_abnormal_line(lines, threshold=0.1):
      slopes = [calculate_slope(line) for line in lines]
      while len(lines) > 0:
          mean = np.mean(slopes)
          diff = [abs(s - mean) for s in slopes]
          idx = np.argmax(diff)
          if diff[idx] > threshold:
              slopes.pop(idx)
              lines.pop(idx)
          else:
              break
      return lines
  
  left_lines = reject_abnormal_line(left_lines)
  right_lines = reject_abnormal_line(right_lines)
  print(len(left_lines), len(right_lines))
  
  # 组内拟合：最小二乘拟合
  def least_squares_fit(lines):
      """
      :param lines: 线段集合 [[x1,y1,x2,y2],...]
      :return: 直线的最小二乘拟合
      """
      # 取出所有坐标点
      x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
      y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
  
      # 进行直线拟合
      ploy = np.polyfit(x_coords, y_coords, deg=1)
  
      # 根据多项式系数 计算直线上的点 唯一确定这条直线
      point_min = (np.min(x_coords),np.polyval(ploy,np.min(x_coords)))
      point_max = (np.max(x_coords),np.polyval(ploy,np.max(x_coords)))
  
      return np.array([point_min, point_max], dtype=np.int32)
  
  left_line = least_squares_fit(left_lines)
  right_line = least_squares_fit(right_lines)
  
  img = cv2.imread('data/170927_063834858_Camera_5.jpg',cv2.IMREAD_COLOR)
  cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), (0, 0, 255), 2)
  cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), (0, 255, 0), 2)
  cv2.imshow('img', img)
  cv2.waitKey(0)
  ```

  































































































