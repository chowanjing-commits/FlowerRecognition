import cv2
import numpy as np
from PIL import ImageGrab
import pygetwindow as gw
from ultralytics  import YOLO
import supervision as sv
import time

# 初始化YOLO模型
model = YOLO(f'best.pt')

# 初始化监督库的注解器
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 目标窗口标题（替换为实际的画图软件窗口标题）
target_window_title = "2.jpg"

while True:
    # 获取目标窗口
    window = None
    windows = gw.getWindowsWithTitle(target_window_title)
    if windows:
        window = windows[0]
    else:
        print(f"找不到窗口: {target_window_title}")
        time.sleep(1)
        continue

    # 获取窗口的位置和尺寸
    x, y, width, height = window.left, window.top, window.width, window.height

    # 截取窗口图像
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    frame = np.array(screenshot)

    # 转换颜色空间从RGB到BGR (OpenCV 使用BGR)
    image_src = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    size_x,size_y= image_src.shape[1], image_src.shape[0]
    image_det = cv2.resize(image_src,(640,640))

    
    result = model.predict(source=image_det,imgsz=640,conf=0.8,save=False)
    boxes=result[0].boxes.xywhn
 
    for box in boxes:
        cv2.rectangle(image_src,(int((box[0]-box[2]/2)* size_x),int((box[1]- box[3]/2)* size_y)),
(int((box[0]+ box[2]/2)*size_x),int((box[1] + box[3]/2)* size_y)),
color=(255,255,0),thickness=2)
    cv2.imshow("frame",image_src)

    if cv2.waitKey(1)== ord('q'):
        break
    pass

    # 使用YOLO模型进行检测
    #results = model(frame)[0]  

    # 从结果字典中提取检测信息（示例代码，具体根据返回的字典结构调整）
    # detections = results.pandas().xywh[0]  # 示例提取边界框信息的方式，具体根据返回的字典结构调整

   
  
 
cv2.destroyAllWindows()
