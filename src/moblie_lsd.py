import cv2
import numpy as np
import tensorflow as tf

# 选择模型：尺寸/精度可根据设备替换
MODEL_PATH = "../model/M-LSD_320_tiny_fp32.tflite"
INPUT_SIZE = 320  # 模型尺寸 must match

def preprocess(img: np.ndarray):
    h, w = img.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    pad = np.zeros((INPUT_SIZE, INPUT_SIZE, 4), dtype=np.uint8)
    pad[:resized.shape[0], :resized.shape[1], :3] = resized
    pad[...,3] = 255  # alpha 通道
    inp = pad.astype(np.float32) / 255.0
    return inp[np.newaxis, ...]

def postprocess(pred: np.ndarray, scale: float, threshold=0.2):
    """
    处理 MLSD 模型输出，提取线段端点。
    输入:
        - pred: 模型输出, shape [1, H, W, 7]
        - scale: 从模型输出空间到原图空间的缩放因子
        - threshold: 线段存在概率阈值
    输出:
        - lines: [(pt1, pt2), ...]
    """
    pred = pred.squeeze()  # [H, W, 7]
    heatmap = pred[..., 0]  # 线段存在概率
    offsets = pred[..., 1:5]  # dx0, dy0, dx1, dy1

    lines = []
    h, w = heatmap.shape
    for y in range(h):
        for x in range(w):
            if heatmap[y, x] > threshold:
                dx0, dy0, dx1, dy1 = offsets[y, x]
                x0 = (x + dx0) / scale
                y0 = (y + dy0) / scale
                x1 = (x + dx1) / scale
                y1 = (y + dy1) / scale
                lines.append(((int(x0), int(y0)), (int(x1), int(y1))))
    return lines


def draw(img, lines):
    vis = img.copy()
    for (p0,p1) in lines:
        cv2.line(vis, p0, p1, (0,255,0), 2)
    return vis

def main():
    img = cv2.imread("../output/cropped_1.png")
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    inp = preprocess(img)

    # TFLite 推理
    interp = tf.lite.Interpreter(MODEL_PATH)
    interp.allocate_tensors()
    input_index = interp.get_input_details()[0]['index']
    output_details = interp.get_output_details()
    interp.set_tensor(input_index, inp)
    interp.invoke()
    # outputs = [interp.get_tensor(o['index'])[0] for o in output_details]
    output_tensor = interp.get_tensor(output_details[0]['index'])  # shape: [1, 320, 320, 7]
    print(f"output_shape: {output_tensor.shape}")
    lines = postprocess(output_tensor, scale=INPUT_SIZE / max(img.shape[:2]))
    vis = draw(img, lines)
    cv2.imshow("MLSD result", vis)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
