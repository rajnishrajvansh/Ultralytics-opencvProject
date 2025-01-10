import cv2
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

model = YOLO('yolov9m.pt')  
image_path = 'crowd.jpg' 
image = cv2.imread(image_path)
output_dir = "cropped_images"
os.makedirs(output_dir, exist_ok=True)
results = model.predict(image)
detection_output = []

for idx, box in enumerate(results[0].boxes, start=1):
    bbox = box.xyxy[0].cpu().numpy() 
    conf = box.conf.cpu().item()      
    cls = int(box.cls.cpu().item())  
    label = model.names[cls]         

    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cropped_object = image[y1:y2, x1:x2]
    cropped_path = os.path.join(output_dir, f"{label}_{idx}.jpg")
    cv2.imwrite(cropped_path, cropped_object)
  
    sub_objects = [] 

    detection_output.append({
        "object": label,
        "id": idx,
        "bbox": [x1, y1, x2, y2],
        "subobject": [
            {
                "object": sub_label, 
                "id": sub_idx,
                "bbox": [sx1, sy1, sx2, sy2]  
            } for sub_idx, (sub_label, sx1, sy1, sx2, sy2) in enumerate(sub_objects, start=1)
        ]
    })

output_json_path = "detection_output_with_subobjects.json"
with open(output_json_path, "w") as json_file:
    json.dump(detection_output, json_file, indent=4)

print(f"Detection output saved as {output_json_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off")
plt.show()

cv2.imwrite("output_image_with_boxes.jpg", image)
print("Processed image saved as output_image_with_boxes.jpg")
