"""色环轮廓的识别"""

from typing import List
import cv2
import numpy as np
import os
import logging as log

# HSV空间阈值
# 数据结构 嵌套字典
# 为了处理像是红色这种分段的情况
# 所以将阈值封装在了数组中, 后续可以直接迭代(这个巧妙)
hsv_thresholds = {
    "blue": {
        "lower": [np.array([95, 80, 80])],
        "upper": [np.array([135, 255, 255])],
    },
    "red": {
        "lower": [np.array([0, 100, 100]), np.array([160, 100, 100])],
        "upper": [np.array([10, 255, 255]), np.array([179, 255, 255])],
    },
    "yellow": {
        "lower": [np.array([20, 100, 100])],
        "upper": [np.array([35, 255, 255])],
    },
}


def clean_mask(
    hsv_image: np.ndarray,
    lower_bounds: List[np.ndarray],
    upper_bounds: List[np.ndarray],
    kernel_size: Tuple[int, int] = (7, 7),
    debug: bool = False,
) -> np.ndarray:
    """通过HSV空间阈值得到轮廓完整, 无外部噪点的单一颜色掩码

    通过HSV空间阈值, 并进行形态学操作,
    得到轮廓完整, 无外部噪点的单一颜色掩码
    特别是消除轮廓之外一些小的噪点

    Args:
        hsv_image (np.ndarray): HSV空间的图像
        lower_bounds (List[np.ndarray]): 下阈值列表
        upper_bounds (List[np.ndarray]): 上阈值列表
        kernel_size (Tuple[int, int]): 形态学操作核大小, 默认为 (7, 7)

    Returns:
        np.ndarray: 单一颜色掩码


    Example:
        image = cv2.imread(<input_image_path>)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bounds = [np.array([95, 80, 80])] # 蓝色下限
        upper_bounds = [np.array([135, 255, 255])] # 蓝色上限
        cleaned_mask = clean_mask(hsv, lower_bounds, upper_bounds, (5, 5))
    """

    # 输入验证
    assert hsv_image.ndim == 3 and hsv_image.shape[2] == 3, "输入必须是HSV图像"
    assert len(lower_bounds) == len(upper_bounds), "阈值列表长度必须匹配"

    # hsv_image 形状为 (height, width, 3)
    # shape[:2] 返回 (height, width)
    # 构建一个全黑掩码
    color_filter_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    # 因为将范围封装在了数组中, 所以可以用zip函数遍历, 非常巧妙
    for lower, upper in zip(lower_bounds, upper_bounds):
        # inRange函数需要 lower 各维度的值都小于 upper 对应维度的值
        # 满足在范围内为1, 不满足为0
        mask_part = cv2.inRange(hsv_image, lower, upper)
        # 由于可能存在多分区阈值的情况，这里需要将所有分区的掩码合并
        # 所以循环将 mask 和 mask_part 进行 or 操作
        color_filter_mask = cv2.bitwise_or(color_filter_mask, mask_part)

    # 可视化基于颜色过滤得到的 mask
    if debug:
        cv2.imshow("mask", color_filter_mask)

    # 使用形态学操作进一步优化轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed_mask = cv2.morphologyEx(color_filter_mask, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    # 基于连通域分析优化轮廓
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened_mask, connectivity=8
    )
    if num_labels <= 1:
        log.warning("未能检测到有效连通域, 返回空掩码")
        return np.zeros_like(opened_mask)
    # 找到最大连通域的索引
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned_mask = np.uint8(labels == largest_label) * 255

    # 可视化优化后的 cleaned_mask
    if debug:
        cv2.imshow("cleaned_mask", cleaned_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cleaned_mask


def main():
    # 读取文件并进行resize操作(原图尺寸较大)
    idx = 2
    input_image_path = f"imgs/img_{idx}.jpg"
    if not os.path.exists(input_image_path):
        print(f"Error: Input file does not exist at '{input_image_path}'.")
        return
    raw_image = cv2.imread(input_image_path)
    if raw_image is None:
        print(f"Error: Could not load image: {input_image_path}")
        return
    # destination size(dsize)
    image = cv2.resize(raw_image, (360, 640), None, None, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_image_with_fit = image.copy()

    # 转换色度空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 遍历得到不同的颜色的外部轮廓
    for color_name, hsv_threshold in hsv_thresholds.items():
        cleaned_mask = clean_mask(hsv, hsv_threshold["lower"], hsv_threshold["upper"])
        # 使用bitwise_and保留掩码区域内容
        masked_image = cv2.bitwise_and(image, image, mask=cleaned_mask)
        cv2.imshow("Result", masked_image)
        cv2.waitKey(0)

        if np.sum(cleaned_mask) == 0:
            print(f" - Could not find the '{color_name}' region.")
            continue

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # 得到轮廓的凸包
            hull = cv2.convexHull(largest_contour)

            # 基于凸包进行椭圆拟合
            if len(hull) >= 5:
                ellipse = cv2.fitEllipse(hull)
                cv2.ellipse(result_image_with_fit, ellipse, (80, 80, 255), 2)

                # 如果是蓝色，构建一个靶面完整mask(用于扣出完整靶面)
                if color_name == "blue":
                    full_target_mask = np.zeros_like(gray, dtype=np.uint8)
                    full_target_mask = (
                        full_target_mask[:, :, 0]
                        if len(full_target_mask.shape) == 3
                        else full_target_mask
                    )  # 确保单通道
                    # 绘制填充椭圆（内部255）
                    cv2.ellipse(full_target_mask, ellipse, color=255, thickness=-1)
                    inverse_mask = cv2.bitwise_not(full_target_mask)  # 按位取反
                    without_target_image = cv2.bitwise_and(
                        image, image, mask=inverse_mask
                    )

                    # 下方通过裁剪得到roi，也可直接使用cv2 roi
                    side_length = 400
                    center_point = tuple(map(int, ellipse[0]))
                    h, w = without_target_image.shape[:2]
                    half_len = side_length // 2
                    # 计算裁剪区域的边界
                    x1 = max(0, center_point[0] - half_len)
                    y1 = max(0, center_point[1] - half_len)
                    x2 = min(w, center_point[0] + half_len)
                    y2 = min(h, center_point[1] + half_len)
                    cropped = image[y1:y2, x1:x2]
                    cv2.imshow("cropped", cropped)
                    cropped_output_path = f"../output/cropped_{idx}.png"
                    if not cv2.imwrite(cropped_output_path, cropped):
                        print(f"Error: Could not write the '{cropped}' image.")
                        return
                    cv2.waitKey(0)
                    cv2.destroyWindow("cropped")

                # Print info
                center = tuple(map(int, ellipse[0]))
                axes = tuple(map(int, ellipse[1]))
                print(
                    f"({color_name.capitalize()}): Center≈{center}, Axes≈({axes[1]:.0f}, {axes[0]:.0f})"
                )

    cv2.destroyWindow("Result")
    output_path_fit = f"../output/{idx}.png"
    if not cv2.imwrite(output_path_fit, result_image_with_fit):
        print(f"Error: Could not write the '{output_path_fit}' image.")
        return

    cv2.imshow("Final Result (Hull Optimized to Ellipse)", result_image_with_fit)
    print("\nPress any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
