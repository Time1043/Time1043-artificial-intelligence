import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

bin_img = cv2.imread('data/170927_063834858_Camera_5_bin.png', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('bin_img', bin_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(bin_img.shape)

# 采集bin_img的坐标点
points = []
for i in range(bin_img.shape[0]):
    for j in range(bin_img.shape[1]):
        if bin_img[i][j] == 255:
            points.append((j, i))
# print(points)

# 通过最小二乘法拟合直线 (可能有多条曲线 需要考虑相关性的问题 即两个点相邻过远就不拟合在一条线上)
def func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(func, [p[0] for p in points], [p[1] for p in points])
print(popt)

# 绘制拟合曲线
x = np.arange(bin_img.shape[1])
y = func(x, *popt)
plt.plot(x, y, color='r')
plt.scatter([p[0] for p in points], [p[1] for p in points], color='b')
plt.show()
