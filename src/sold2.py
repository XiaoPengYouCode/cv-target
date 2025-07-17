import cv2
import torch
import kornia as K
import kornia.feature as KF
import numpy as np


def load_image_as_gray_tensor(image_path: str) -> torch.Tensor:
    """从本地加载图片为灰度张量，shape = [1, 1, H, W]"""
    img = K.io.load_image(image_path, K.io.ImageLoadType.RGB32)
    img = img.unsqueeze(0)  # 添加 batch 维度
    img_gray = K.color.rgb_to_grayscale(img)
    return img_gray


def draw_lines_opencv(
    image_bgr: np.ndarray, lines: np.ndarray, color=(0, 255, 255)
) -> np.ndarray:
    """在 OpenCV 图像上绘制线段（BGR）"""
    image_with_lines = image_bgr.copy()
    for line in lines:
        pt1 = tuple(np.round(line[0][::-1]).astype(int))  # (x0, y0)
        pt2 = tuple(np.round(line[1][::-1]).astype(int))  # (x1, y1)
        cv2.line(image_with_lines, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
    return image_with_lines


def filter_by_length(lines: np.ndarray, min_length: float) -> np.ndarray:
    """保留长度大于 min_length 的线段"""
    keep = []
    for line in lines:
        p1, p2 = line
        length = np.linalg.norm(p1 - p2)
        if length >= min_length:
            keep.append(line)
    return np.array(keep)


def main():
    # 替换为你的图片路径
    image_path = "../output/cropped_2.png"

    # 加载图像为灰度 Tensor
    img_gray = load_image_as_gray_tensor(image_path)

    # 初始化 SOLD2 模型（使用预训练模型）
    sold2 = KF.SOLD2(pretrained=True)

    # 检测线段
    with torch.inference_mode():
        outputs = sold2(img_gray)
    lines = outputs["line_segments"][0].cpu().numpy()  # shape: [N, 2, 2]

    suitable_lines = filter_by_length(lines, 300)

    # 用 OpenCV 加载原图（BGR）
    image_bgr = cv2.imread(image_path)

    # 绘制线段
    image_with_lines = draw_lines_opencv(image_bgr, lines)

    # 显示结果
    cv2.imshow("Detected Lines", image_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
