import cv2
import pytesseract
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\harbourshield-firebase-adminsdk-xh1az-970e36e399.json")  # Update with your service account key
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your Tesseract path

def extract_text_from_image(image):
    # Perform OCR on the image
    text = pytesseract.image_to_string(image, config='--psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return text.strip()  # Strip whitespace from the extracted text

def send_text_to_firestore(text):
    if text:  # Check if text is not empty
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Send text data along with current time to Firestore
        doc_ref = db.collection('OCR').document()
        doc_ref.set({
            'text': text,
            'timestamp': current_time
        })
        print("Text data sent to Firestore.")
    else:
        print("No text detected in the image. Not sending to Firestore.")

cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    text = extract_text_from_image(frame)
    print("Detected Text:", text)
    send_text_to_firestore(text)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
