import cv2
import face_recognition
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import base64

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\harbourshield-firebase-adminsdk-xh1az-970e36e399.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://harbourshield-default-rtdb.firebaseio.com/'})
database_ref = db.reference('/detected_people')

video_capture = cv2.VideoCapture(0)
# Load known face encodings and names
naziya_image = face_recognition.load_image_file(r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\Naziya.jpeg")
naziya_encoding = face_recognition.face_encodings(naziya_image)[0]

sanya_image = face_recognition.load_image_file(r"C:\Users\naziy\OneDrive\Desktop\HarbourShield Python\Sanya.jpeg")
sanya_encoding = face_recognition.face_encodings(sanya_image)[0]

known_face_encoding = [naziya_encoding, sanya_encoding]
known_faces_names = ["Naziya", "Sanya"]

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    detected_info = []  # Reset the detected information for each frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        # Extract the face region from the frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        face_image = frame[top:bottom, left:right]

        # Encode the face image to base64
        _, buffer = cv2.imencode('.jpg', face_image)
        face_bytes = buffer.tobytes()
        face_base64 = base64.b64encode(face_bytes).decode('utf-8')

        detected_info.append({
            "name": name,
            "dateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "face_image": face_base64
        })

        if matches[best_match_index]:
            rectangle_color = (0, 255, 0)  # Green for known faces
        else:
            rectangle_color = (0, 0, 255)  # Red for unknown faces

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)

        # Display the name
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (left, bottom + 30)
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 2
        line_type = 2

        cv2.putText(frame, name, bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)

    # Store detected information in Firebase Realtime Database
    if detected_info:  # Only update Firebase if there's detected information
        data_to_push = {
            "detected_info": detected_info
        }
        database_ref.push().set(data_to_push)

    # Display the live video locally
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
