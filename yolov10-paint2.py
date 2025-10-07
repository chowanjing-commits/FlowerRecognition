import cv2
import numpy as np
from PIL import ImageGrab
import pygetwindow as gw
from ultralytics import YOLO
import supervision as sv
import time
import os
import torch

# ==================== 配置参数 ====================
CONFIG = {
    "model_path": "best.pt",               # 模型路径
    "target_window_title": "2.jpg",        # 目标窗口标题（严格匹配窗口标题）
    "display_window_name": "Detection",    # 显示窗口名称
    "display_window_size": (800, 600),     # 显示窗口初始大小
    "detection_interval": 0.01,            # 检测间隔（秒）
    "inference_device": "cuda" if torch.cuda.is_available() else "cpu",  # 推理设备
    "inference_imgsz": 640                 # 推理输入尺寸
}

# ==================== 初始化组件 ====================
def init_components():
    if not os.path.exists(CONFIG["model_path"]):
        raise FileNotFoundError(f"模型文件不存在: {CONFIG['model_path']}")
    
    try:
        model = YOLO(CONFIG["model_path"])
        model.to(CONFIG["inference_device"])
        print(f"模型加载成功，设备: {CONFIG['inference_device']}")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")
    
    # 优化标注器样式（更清晰）
    bounding_box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.ColorPalette.DEFAULT
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.7,
        text_thickness=2,
        text_color=sv.Color.WHITE
    )
    
    return model, bounding_box_annotator, label_annotator

# ==================== 核心功能函数 ====================
def get_target_window(title):
    """获取目标窗口，返回窗口对象或None"""
    windows = gw.getWindowsWithTitle(title)
    return windows[0] if windows else None

def capture_window(window):
    """截取完整窗口图像，修复坐标问题"""
    # 强制转换为整数坐标（避免小数导致截图偏移）
    x = int(window.left)
    y = int(window.top)
    width = int(window.width)
    height = int(window.height)
    
    # 检查窗口是否在屏幕范围内
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        print(f"窗口坐标异常: x={x}, y={y}, width={width}, height={height}")
        return None
    
    # 截图时扩展1像素避免截断（部分窗口边框可能被裁剪）
    try:
        screenshot = ImageGrab.grab(bbox=(x, y, x + width + 1, y + height + 1))
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"截图失败: {str(e)}")
        return None

# ==================== 主程序 ====================
def main():
    try:
        model, bounding_box_annotator, label_annotator = init_components()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    cv2.namedWindow(CONFIG["display_window_name"], cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG["display_window_name"], *CONFIG["display_window_size"])
    
    prev_time = 0
    fps = 0
    
    print("开始检测...（按ESC退出）")
    while True:
        # 1. 获取目标窗口
        window = get_target_window(CONFIG["target_window_title"])
        if not window:
            print(f"找不到窗口: {CONFIG['target_window_title']}，重试中...")
            time.sleep(1)
            continue
        
        # 2. 截取完整窗口图像
        frame = capture_window(window)
        if frame is None:
            time.sleep(CONFIG["detection_interval"])
            continue
        
        # 3. 目标检测
        results = model(
            frame,
            imgsz=CONFIG["inference_imgsz"],
            device=CONFIG["inference_device"],
            verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 4. 标注图像
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        # 5. 计算并显示FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(
            annotated_image,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # 6. 打印检测结果
        det_count = len(detections)
        print(f"检测到 {det_count} 个目标 | FPS: {int(fps)}")
        
        # 7. 显示图像（检查窗口状态）
        if cv2.getWindowProperty(CONFIG["display_window_name"], cv2.WND_PROP_VISIBLE) <= 0:
            print("窗口已关闭，退出程序")
            break
        cv2.imshow(CONFIG["display_window_name"], annotated_image)
        
        # 8. 退出逻辑
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC键
            print("ESC键按下，退出程序")
            break
        
        time.sleep(CONFIG["detection_interval"])
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()