import cv2
import numpy as np
import tensorflow as tf
import os
import pathlib


data_directory = pathlib.Path('D:/college/Python/github/deep-learning/projects/SignLanguageRecognition/dataset-87000/asl_alphabet_train');
folder_names = [name for name in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, name))]
folder_names = sorted(folder_names, key=lambda x: (x[0].islower(), x))
alphabet_labels_dict = {
    folder: index for index, folder in enumerate(folder_names)
}
alphabets_dict = {value: key for key, value in alphabet_labels_dict.items()}

model = tf.keras.models.load_model('D:/college/Python/github/deep-learning/projects/SignLanguageRecognition/SignLanguageRecognition-87000.keras')

cap = cv2.VideoCapture(0)

desired_width = 1920
desired_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

square_size = 400   #Side length of the square
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x_offset = -300
x_start = (frame_width - square_size) // 2 + x_offset   #Center the square horizontally
y_start = (frame_height - square_size) // 2   #Center the square vertically

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #Draw a square on the frame
    cv2.rectangle(frame, (x_start, y_start), (x_start + square_size, y_start + square_size), (0, 255, 0), 2)
    
    #Extract the square region of interest (ROI)
    square_roi = frame[y_start:y_start + square_size, x_start:x_start + square_size]
    
    input_frame = cv2.resize(square_roi, (64,64))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    
    prediction = model.predict(input_frame)
    predicted_class = np.argmax(prediction)
    
    cv2.putText(frame, f"Prediction: {alphabets_dict[predicted_class]}",(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("ASL Detection", frame)
    
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()