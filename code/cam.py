import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.utils import img_to_array


img_file_buffer = st.camera_input("Take a picture")
model=tf.keras.models.load_model('../Saved-Models/model4829.h5')
classifier = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
activiteis = ["Home", "Webcam Face Detection","Picture Expression Classification", "Image Cropper", "User Manual"]
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=cv2_img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            predictions = model.predict(roi)
            max_index = int(np.argmax(predictions))
        label_position = (x, y)
        result = str(emotions[max_index]) + " " + str(np.round(np.max(predictions[0]), 2))
        al = cv2.putText(cv2_img, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        st.image(al)