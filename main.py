import cv2
from deepface import DeepFace
import numpy as np

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not video.isOpened():
    print('Could not open video source')
else:
    print('Opened video source')

side_image = cv2.imread("./content/neutral.jpg")
side_image = cv2.resize(side_image, (400, 480))

frame_width = 640 + 400
frame_height = 480

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

frame_count = 0

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    if frame_count % 10 != 0:
        combined = np.hstack((frame, side_image))
        out.write(combined)
        cv2.imshow('Camera', combined)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        try:
            analyze = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analyze[0]['dominant_emotion'] if isinstance(analyze, list) else analyze['dominant_emotion']
            try:
                img_path = f'./content/{dominant_emotion}.jpg'
                side_image = cv2.imread(img_path)
            except Exception as e:
                print(e)

        except Exception as e:
            print("Error detecting face:", e)

    side_image = cv2.resize(side_image, (400, 480))

    combined = np.hstack((frame, side_image))
    out.write(combined)
    cv2.imshow('Camera', combined)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()