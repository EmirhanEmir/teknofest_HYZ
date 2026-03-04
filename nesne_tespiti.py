import os
import cv2
from ultralytics import YOLO

import torch
print(f"CUDA Mevcut mu: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Cihaz Adı: {torch.cuda.get_device_name(0)}")

font = cv2.FONT_HERSHEY_SIMPLEX
img_folder_path = r"Frames_jpg"
model_path = r"Models/best.pt"

model = YOLO(model_path)
threshold = 0.6


window_name = "Yolo v11"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Pencereyi yeniden boyutlandırılabilir yap
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Tam ekran yap

for img_path in os.listdir(img_folder_path):
    img_path = os.path.join(img_folder_path, img_path)
    img = cv2.imread(img_path)

    if img is None: # görüntü yoksa geç
        continue

    detect_result = model(img)[0] # çoklu tespit için yazlımış manuel [0] al
    result = detect_result.boxes.data.tolist() # sonuç içindeki listeler

    if len(result) > 0:
        
        x1,y1,x2,y2,score,class_id = result[0] # liste yapısında olduğu için mauel [0] al
        x1,y1,x2,y2,class_id = int(x1),int(y1),int(x2),int(y2),int(class_id)

        if score > threshold:
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            class_name = detect_result.names[class_id]
            score = score*100

            text = f"{class_name}:%{score:.2f}"

            cv2.putText(img, text, (x1,y1-10), font, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Yolo v11", img)
    key = cv2.waitKey(1)
    if key==27: # esc tuşu
        break

cv2.destroyAllWindows()
