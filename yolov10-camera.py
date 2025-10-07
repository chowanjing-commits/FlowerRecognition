import cv2
import supervision as sv
from ultralytics import YOLOv10

# 1. 模型加载（添加异常处理，避免启动失败）
try:
    model = YOLOv10("best.pt")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 2. 初始化标注器（可保持原配置）
bounding_box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=2  # 注意：10px 较粗，可根据需求调整为 2-5
)
label_annotator = sv.LabelAnnotator()

# 3. 摄像头初始化（检查设备是否正常）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("无法打开摄像头")
    exit(1)

# 4. 创建窗口（仅创建1次，避免重复初始化）
cv2.namedWindow("Webcam", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)

# 5. 主循环（添加窗口状态检测+异常捕获）
try:
    while True:
        # 读取摄像头帧（检查帧是否有效）
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧，退出循环")
            break

        # 模型推理与标注
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # 显示图像（关键：先检查窗口是否存在，再显示）
        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) > 0:
            cv2.imshow("Webcam", annotated_image)
        else:
            # 窗口已被手动关闭，中断循环
            print("窗口已手动关闭，退出程序")
            break

        # 监听退出事件（ESC键或'q'键）
        k = cv2.waitKey(1)
        if k % 256 == 27 or k == ord('q'):  # 27=ESC，ord('q')=字母q
            print("Escape/q键触发，关闭程序")
            break

except Exception as e:
    # 捕获循环中意外错误（如摄像头断开）
    print(f"程序运行异常: {e}")

finally:
    # 6. 强制释放资源（无论何种退出方式，都确保关闭）
    cap.release()  # 释放摄像头
    cv2.destroyWindow("Webcam")  # 强制销毁指定窗口
    cv2.destroyAllWindows()  # 兜底：销毁所有残留窗口