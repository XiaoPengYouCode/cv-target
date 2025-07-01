"""色环轮廓的识别"""

import cv2
import numpy as np
import os

# HSV空间阈值
color_definitions = {
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
    }
}

def create_clean_mask(hsv_image, lower_bounds, upper_bounds):
    """
    通过HSV空间阈值，得到单一颜色掩码
    """
    final_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in zip(lower_bounds, upper_bounds):
        mask_part = cv2.inRange(hsv_image, lower, upper)
        final_mask = cv2.bitwise_or(final_mask, mask_part)

    # 使用形态学操作进一步优化轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(opened_mask)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean_mask = np.uint8(labels == largest_label) * 255

    return clean_mask


def main():
    # 读取文件并进行resize操作(原图尺寸较大)
    idx = 2
    input_image_path = f'../imgs/img_{idx}.jpg'
    if not os.path.exists(input_image_path):
        print(f"Error: Input file does not exist at '{input_image_path}'.")
        return
    raw_image = cv2.imread(input_image_path)
    if raw_image is None:
        print(f"Error: Could not load image: {input_image_path}")
        return
    image = cv2.resize(raw_image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_image_with_fit = image.copy()

    # 转换色度空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 遍历得到不同的颜色的外部轮廓
    for color_name, definition in color_definitions.items():
        clean_mask = create_clean_mask(hsv, definition["lower"], definition["upper"])
        # 使用bitwise_and保留掩码区域内容
        result = cv2.bitwise_and(image, image, mask=clean_mask)
        cv2.imshow("Result", result)
        cv2.imshow('Final Result (Hull Optimized to Ellipse)', clean_mask)
        cv2.waitKey(0)

        if np.sum(clean_mask) == 0:
            print(f" - Could not find the '{color_name}' region.")
            continue

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # 得到轮廓的凸包
            hull = cv2.convexHull(largest_contour)

            # 基于凸包进行椭圆拟合
            if len(hull) >= 5:
                ellipse = cv2.fitEllipse(hull)
                cv2.ellipse(result_image_with_fit, ellipse, (80, 80, 255), 2)

                # 如果是蓝色，构建一个靶面完整mask(用于扣出完整靶面)
                if color_name == 'blue':
                    full_target_mask = np.zeros_like(gray, dtype=np.uint8)
                    full_target_mask = full_target_mask[:, :, 0] if len(full_target_mask.shape) == 3 else full_target_mask  # 确保单通道
                    # 绘制填充椭圆（内部255）
                    cv2.ellipse(full_target_mask, ellipse, color=255, thickness=-1)
                    inverse_mask = cv2.bitwise_not(full_target_mask)  # 按位取反
                    without_target_image = cv2.bitwise_and(image, image, mask=inverse_mask)

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
                    cv2.imshow('cropped', cropped)
                    cropped_output_path = f'../output/cropped_{idx}.png'
                    if not cv2.imwrite(cropped_output_path, cropped):
                        print(f"Error: Could not write the '{cropped}' image.")
                        return
                    cv2.waitKey(0)
                    cv2.destroyWindow('cropped')

                # Print info
                center = tuple(map(int, ellipse[0]))
                axes = tuple(map(int, ellipse[1]))
                print(f"({color_name.capitalize()}): Center≈{center}, Axes≈({axes[1]:.0f}, {axes[0]:.0f})")

    cv2.destroyWindow('Result')
    output_path_fit = f'../output/{idx}.png'
    if not cv2.imwrite(output_path_fit, result_image_with_fit):
        print(f"Error: Could not write the '{output_path_fit}' image.")
        return

    cv2.imshow('Final Result (Hull Optimized to Ellipse)', result_image_with_fit)
    print("\nPress any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()