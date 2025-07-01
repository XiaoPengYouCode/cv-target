import cv2
import numpy as np


def detect_corners(image) -> np.array:
    """
    检测图像中的主要四边形（例如文档）的四个角点，并将其排序为
    左上、右上、右下、左下的顺序。

    Args:
        image (np.array): 输入图像 (OpenCV格式，BGR)。

    Returns:
        np.array: 包含四个角点坐标的NumPy数组，顺序为 [左上, 右上, 右下, 左下]。
                  如果未检测到四边形，则返回图像的四个角点。
    """
    # 调整图像大小 (可选)。
    # 默认 scale_percent 为 100，意味着不缩放。
    # 如果图片很大，适当缩小可以提高处理速度。
    scale_percent = 100  # 调整缩放百分比，例如 50 表示缩小一半
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    # 确保 width 和 height 至少为1，避免resize报错
    width = max(1, width)
    height = max(1, height)
    image_resized = cv2.resize(image, (width, height))

    # 转换为灰度图
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # 高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's 二值化
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积降序排序轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page_contour = None
    # 遍历轮廓以找到页面的轮廓 (假设是最大的四边形)
    for contour in contours:
        # 逼近轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)  # 0.02 是经验值，可能需要微调
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果逼近后的轮廓有4个点，我们假定它是页面
        if len(approx) == 4:
            # 可以添加额外的筛选条件，例如最小面积或长宽比，以避免噪声
            area = cv2.contourArea(approx)
            # 这里的1000是一个经验值，可能需要根据您的图片大小调整
            if area > 1000:
                page_contour = approx
                break

    # 如果没有找到符合条件的四边形，返回图像的四个角点
    if page_contour is None:
        print("未检测到符合条件的四边形页面。返回图像的默认角点。")
        return np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )

    # 提取角点坐标
    # page_contour 的形状通常是 (4, 1, 2)，我们需要将其扁平化为 (4, 2)
    corner_points = page_contour[:, 0, :].astype(np.float32)

    # 计算轮廓的中心点，用于排序角点
    center_x, center_y = np.mean(corner_points, axis=0)

    # 按照相对于中心的角度排序角点，以确保顺序一致性
    # 通常的顺序是：左上、右上、右下、左下
    # atan2 返回的角度范围是 (-pi, pi]，所以需要额外处理以确保正确的顺序
    # 简单的方法是按 x+y 排序，然后按 x-y 排序，或者更可靠的：
    # 1. 找到左上角 (x+y 最小)
    # 2. 找到右下角 (x+y 最大)
    # 3. 剩下两个点，找到左下角 (x-y 最小), 右上角 (x-y 最大)

    # 另一种更可靠的排序方法（来自OpenCV文档或其他标准实现）
    # 初始化一个存储排序后角点的数组
    # corners[0] = top-left, corners[1] = top-right
    # corners[2] = bottom-right, corners[3] = bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # 左上角 (sum 最小) 和 右下角 (sum 最大)
    s = corner_points.sum(axis=1)
    rect[0] = corner_points[np.argmin(s)]
    rect[2] = corner_points[np.argmax(s)]

    # 右上角 (diff 最小) 和 左下角 (diff 最大)
    diff = np.diff(corner_points, axis=1)
    rect[1] = corner_points[np.argmin(diff)]
    rect[3] = corner_points[np.argmax(diff)]

    return rect



# 辅助函数，封装原有的detect_corners逻辑，并返回处理后的图像
def _detect_corners_and_return_image(image) -> tuple[np.array, np.array]:
    # Resize the image for better processing (optional)
    scale_percent = 100  # Adjust the percentage for resizing
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    width = max(1, width)
    height = max(1, height)
    image_resized = cv2.resize(image, (width, height))

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page_contour = None
    # Loop through contours to find the page contour
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximated contour has 4 points, we assume it's the page
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 1000:  # 面积筛选，确保不是太小的噪声
                page_contour = approx
                break
    else:
        print("未检测到符合条件的四边形页面。")
        # 返回图像的默认角点，以及原始调整大小后的图像
        default_corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )
        return default_corners, image_resized

    # Refine corner detection using perspective transformation
    corner_points = page_contour[:, 0, :].astype(np.float32)

    # Sort points consistently: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = corner_points.sum(axis=1)
    rect[0] = corner_points[np.argmin(s)]  # top-left has smallest sum (x+y)
    rect[2] = corner_points[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(corner_points, axis=1)
    rect[1] = corner_points[np.argmin(diff)]  # top-right has smallest diff (x-y)
    rect[3] = corner_points[np.argmax(diff)]  # bottom-left has largest diff

    # 返回排序后的角点和处理后的图像，以便外部绘制
    return rect, image_resized


# --- 主程序部分 ---
if __name__ == "__main__":
    # 请替换为您的图片路径
    image_path = "../output/cropped_1.png"

    # 1. 加载图片
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"错误：无法加载图片 {image_path}。请检查路径。")
    else:
        # 2. 调用角点检测函数
        # 这里为了在原始图像上显示结果，我们将原始图像的一个副本传递给detect_corners
        # 并在函数内部进行缩放（如果 scale_percent < 100）
        # 或者，您可以在调用前就resize original_image

        # 假设您希望在原图大小上绘制，或者如果detect_corners内部有resize，
        # 则需要根据detect_corners返回的图片大小来绘制。
        # 为了简洁，我们让detect_corners处理resize，并返回原始图像上对应的坐标。

        # 为了在原始尺寸图像上绘制，我们需要将 `detect_corners` 内部的 `image_resized` 返回或者在外部处理
        # 调整 detect_corners 函数的返回，使其只返回坐标，不处理绘图，绘图在外部完成

        # 重新设计一下：detect_corners只负责返回角点，绘图逻辑放在外面

        # 获取原始图像的副本以进行绘制
        image_for_display = original_image.copy()

        # 调用函数获取角点 (注意：函数内部可能会对图像进行resize，所以返回的坐标是resize后的坐标)
        # 如果 detect_corners 内部 resize 了，并且您想在 resize 后的图上绘制，
        # 那么需要将 resize 后的图传出来。
        # 最简单的方式是让 detect_corners 内部不绘制，只返回角点坐标。

        # 重新调用并获取调整后的图像和角点

        # 为了让绘制与返回的角点匹配，我们需要知道 detect_corners 内部是否进行了缩放。
        # 考虑到 `detect_corners` 函数已经包含了 `resize` 逻辑，
        # 并且您的需求是在图片上显示计算得到的四角，
        # 我将让 `detect_corners` 返回处理后的图像和排序后的角点。

        # 优化 detect_corners 函数，使其返回处理后的图像和角点
        corners, processed_image_for_display = _detect_corners_and_return_image(original_image.copy())

        # 在处理后的图像上绘制检测到的轮廓和角点
        if corners is not None and processed_image_for_display is not None:
            # 绘制轮廓 (注意：这里的corners已经是排序好的)
            # 由于 corners 是 (4, 2) 的 float32 数组，需要将其转换为 (4, 1, 2) 的 int32 才能用于 drawContours
            page_contour_to_draw = np.array([corners], dtype=np.int32)
            cv2.drawContours(processed_image_for_display, [page_contour_to_draw], -1, (0, 255, 0), 2)

            # 突出显示角点
            for point in corners:
                x, y = int(point[0]), int(point[1])
                cv2.circle(processed_image_for_display, (x, y), 5, (0, 0, 255), -1)

            print("检测到的角点 (左上, 右上, 右下, 左下):")
            for i, p in enumerate(corners):
                print(f"角点 {i + 1}: ({p[0]:.2f}, {p[1]:.2f})")

            # 显示结果
            cv2.imshow("Detected Corners", processed_image_for_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("角点检测失败。")
