import cv2 as cv
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\harbourshield-firebase-adminsdk-xh1az-970e36e399.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load YOLO
net = cv.dnn.readNet(r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\yolomodel\yolov3.weights",
                     r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\yolomodel\yolov3 (3).cfg")
classes = []
with open(r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\yolomodel\coco (1).names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load Image
img = cv.imread(r"C:\Users\naziy\Downloads\WhatsApp Image 2024-05-29 at 3.42.01 PM.jpeg")
img = cv.resize(img, None, fx=0.4, fy=0.4)
height, width, channel = img.shape

# Detect Objects
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layer)

# Showing Information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] == 'boat':
            # Object detection
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
detected_objects = []

font = cv.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y + 30), font, 3, color, 3)
        detected_objects.append({
            "label": label,
            "confidence": confidences[i],
            "box": {"x": x, "y": y, "w": w, "h": h}
        })

# Display the image
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Send the detected objects to Firebase
doc_ref = db.collection("detected_ships").add({
    "image_path": r"C:\Users\naziy\Downloads\WhatsApp Image 2024-05-29 at 3.42.01 PM.jpeg",
    "objects": detected_objects
})

print("Data sent to Firebase")
