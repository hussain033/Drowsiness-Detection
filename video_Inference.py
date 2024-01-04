#Importing libraries
import cv2
import numpy as np
import tensorflow as tf

# function to detect faces and then detect eyes within the face using HAARCASCADES
def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    eyes_coordinates = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = frame[y:y+h, x:x+w]
            eyes_coordinates.append((x + ex, y + ey, ew, eh))

    return eyes_coordinates

# function to detect drowiness
def detect_drowsiness(model, eye):
    eye_resized = cv2.resize(eye, (300,300))
    eye_resized = eye_resized.astype(np.float32) / 255.0
    eye_resized = np.expand_dims(eye_resized, axis=0)

    prediction = model.predict(eye_resized)

    drowsiness_score = prediction[0][0]
    print(drowsiness_score)
    if drowsiness_score<0:
        return True
    else:
        return False

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Loading the trained TensorFlow model
    model = tf.keras.models.load_model("model")
    
    input_video_path = 'input.mp4'
    output_video_path = 'output_video.avi'

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    blink_counter = 0
    drowsy_time_threshold = 20  # Set this value based on your requirement (in frames)
    drowsy = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        eyes_coordinates = detect_eyes(frame, face_cascade, eye_cascade)

        drowsy_count = 0
        for (ex, ey, ew, eh) in eyes_coordinates:
            eye = frame[ey:ey+eh, ex:ex+ew]

            drowsy = detect_drowsiness(model, eye)

            if drowsy:
                color = (0, 0, 255)  # Red color for drowsy eyes
                drowsy_count += 1
            else:
                color = (0, 255, 0)  # Green color for alert eyes

            # Draw a rectangle around the detected eyes
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)


        if drowsy_count > 0:
            blink_counter += 1
        else:
            blink_counter = 0

        # Check if the eyes have been closed for a long time (drowsy)
        if blink_counter >= drowsy_time_threshold:
            cv2.putText(frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("Inference complete")
if __name__ == "__main__":
    main()


    