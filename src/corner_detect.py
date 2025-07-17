import cv2
import numpy as np

# 读取图像
image_path = "../output/cropped_1.png"  # 替换为你的图片路径
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("图片路径错误，请检查文件是否存在！")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # (5, 5) 是高斯核大小，0 是标准差
# 4. 应用 Canny 边缘检测
# 参数说明：
# - 第一个参数：输入图像
# - 第二个参数：低阈值
# - 第三个参数：高阈值
# - 阈值的选择会影响边缘检测的效果
edges = cv2.Canny(blurred, threshold1=3, threshold2=10)
cv2.imshow("edges", edges)

# 定义结构元素
kernel = np.ones((5, 5), np.uint8)  # 3x3 的矩形结构元素
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed", closed)
blurred_closed = cv2.GaussianBlur(closed, (3, 3), 0)  # (5, 5) 是高斯核大小，0 是标准差
cv2.imshow("blurred_closed", blurred_closed)

# 3. 腐蚀操作
eroded = cv2.erode(closed, kernel, iterations=1)  # 腐蚀 1 次
cv2.imshow("eroded", eroded)

# 4. 膨胀操作
dilated = cv2.dilate(edges, kernel, iterations=1)  # 膨胀 1 次
cv2.imshow("dilated", dilated)

edges_2 = cv2.Canny(dilated, threshold1=6, threshold2=18)
cv2.imshow("edges_2", edges_2)

# 2. 应用霍夫变换检测直线
lines = cv2.HoughLinesP(
    edges_2, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50
)
# 4. 绘制检测到的直线
if lines is not None:
    print(f"number of lines: {len(lines)}")
    for line in lines:
        x1, y1, x2, y2 = line[0]  # 获取直线的两个端点
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在原图上绘制绿色直线

cv2.imshow("lines", image)

# 等待用户按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. 保存结果（可选）
output_path = "edges_output.jpg"
cv2.imwrite(output_path, edges)
print(f"边缘检测结果已保存到 {output_path}")
