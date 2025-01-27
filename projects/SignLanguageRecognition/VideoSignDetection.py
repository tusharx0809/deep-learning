import cv2
import numpy as np
import tensorflow as tf
import os
import pathlib

# Load model and prepare labels
data_directory = pathlib.Path('D:/college/Python/github/deep-learning/projects/SignLanguageRecognition/dataset/asl_alphabet_train')
folder_names = [name for name in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, name))]
folder_names = sorted(folder_names, key=lambda x: (x[0].islower(), x))
alphabet_labels_dict = {folder: index for index, folder in enumerate(folder_names)}
alphabets_dict = {value: key for key, value in alphabet_labels_dict.items()}

model = tf.keras.models.load_model('D:/college/Python/github/deep-learning/projects/SignLanguageRecognition/SignLanguageRecognition.keras')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

square_size = 400
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x_offset = -350
x_start = (frame_width - square_size) // 2 + x_offset
y_start = (frame_height - square_size) // 2

confidence_threshold = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw square and extract ROI
    cv2.rectangle(frame, (x_start, y_start), (x_start + square_size, y_start + square_size), (0, 255, 0), 2)
    square_roi = frame[y_start:y_start + square_size, x_start:x_start + square_size]
    
    # Preprocess ROI for prediction
    square_roi = cv2.cvtColor(square_roi, cv2.COLOR_BGR2RGB)  # Ensure RGB input for the model
    input_frame = cv2.resize(square_roi, (64, 64))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    
    # Make prediction
    prediction = model.predict(input_frame)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    if confidence > confidence_threshold:
        prediction_text = f"Prediction: {alphabets_dict[predicted_class]} ({confidence:.2f})"
    else:
        prediction_text = "Prediction: Uncertain"
    
    cv2.putText(frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("ASL Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
