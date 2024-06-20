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