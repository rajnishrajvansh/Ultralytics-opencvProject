import cv2
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')  # YOLOv8 Nano model (lightweight and fast)

# Load an image
image_path = 'input_image1.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# Perform object detection
results = model.predict(image)

# Prepare the JSON output structure
detection_output = []

# Process detected objects
for idx, box in enumerate(results[0].boxes, start=1):
    bbox = box.xyxy[0].cpu().numpy()  # Bounding box coordinates [x1, y1, x2, y2]
    conf = box.conf.cpu().item()      # Confidence score
    cls = int(box.cls.cpu().item())   # Class ID
    label = model.names[cls]          # Class name

    # Draw the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add to JSON output
    detection_output.append({
        "object": label,
        "id": idx,
        "bbox": [x1, y1, x2, y2],
        "subobject": None  # Placeholder for sub-objects if any
    })

# Save JSON output to a file
output_json_path = "detection_output2.json"
with open(output_json_path, "w") as json_file:
    json.dump(detection_output, json_file, indent=4)

print(f"Detection output saved as {output_json_path}")

# Convert BGR to RGB for visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

# Save the processed image
cv2.imwrite("output1.jpg", image)
print("Processed image saved as output_image.jpg")
