import os
import streamlit as st
import tensorflow as tf
# from tensorflow import keras
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import math
import time
from streamlit_cropper import st_cropper
from PIL import Image

@st.cache_resource(ttl=10800) # 3小时过期
def load_h5model():
    model = tf.keras.models.load_model('Saved-Models/model7509.h5')
    return model

@st.cache_resource(ttl=10800) # 3小时过期
def load_cv():
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return classifier

@st.cache_resource(ttl=10800) # 3小时过期
def load_RTC():
    # RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    # RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]})
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.ideasip.com:3478"]}]})

    return RTC_CONFIGURATION


model = load_h5model()
classifier = load_cv()
RTC_CONFIGURATION = load_RTC()

activiteis = ["Home", "Webcam Face Detection","Picture Expression Classification", "Image Cropper", "User Manual"]
choice = st.sidebar.selectbox("Select Activity", activiteis)
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
color_RGB = [(128, 128, 128), (255, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0),  (0, 128, 0), (128, 0, 128), (255, 165, 0)]
color_BGR = [(128, 128, 128),(0, 255, 255),(255, 255, 0), (255, 0, 0),(0, 0, 255),(0, 0,128), (128, 0,128),(0,165,255)]
class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:

            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                predictions = model.predict(roi)
                max_index = int(np.argmax(predictions))
            label_position = (x, y)
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=color_BGR[max_index], thickness=2)
            result = str(emotions[max_index]) + " " + str(np.round(np.max(predictions[0]), 2))
            cv2.putText(img, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_BGR[max_index], 2)
        return img


if choice == "Home":
    html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                        <h4 style="color:white;text-align:center;">
                                        Face Expression Identification</h4>
                                        </div>
                                        </br>"""
    st.markdown(html_temp_home1, unsafe_allow_html=True)
    st.write("""Identifying the Face Expression""")
if choice == "Picture Expression Classification":
    img_show = []
    emo_show = []
    uploaded_files = st.sidebar.file_uploader("Choose a Picture jpg/png", accept_multiple_files=True)
    num_files = len(uploaded_files) if uploaded_files is not None else 0
    i = 0
    scale_factor = st.sidebar.slider('Scale Factor', 1.0, 2.0, 1.3)
    min_neighbour = st.sidebar.slider('Min Neighbour', 3, 6, 5)
    start_time = time.time()
    for uploaded_file in uploaded_files:
        if i == 0:
            progress_bar = st.progress(0, text="Operation in progress. Please wait.")
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_resize = np.expand_dims(img, axis=-1)
        faces_detected = classifier.detectMultiScale(img_resize, scale_factor, min_neighbour)
        if len(faces_detected) == 0:
            st.error("No Face Detected")
            st.image(uploaded_file, caption="Not Face", use_column_width=True)
            break;
        j = 0
        for (x, y, w, h) in faces_detected:

            roi_gray = img_resize[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.expand_dims(roi_gray, axis=0)
            img_pixels = img_pixels / 255.0
            # 获取当前时间

            predictions = model.predict(img_pixels)
            # 获取预测完成后的时间

            max_index = int(np.argmax(predictions))
            predicted_emotion = emotions[max_index]
            result = str(predicted_emotion) + " " + str(np.round(np.max(predictions[0]), 2))
            cv2.rectangle(image, (x, y), (x + w, y + h), color_BGR[max_index], 2)
            img_predict = cv2.putText(image, result, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_BGR[max_index], 2)
            j = j + 1
            if j == len(faces_detected):
                img_show.append(img_predict)
                emo_show.append(predicted_emotion)
                #st.image(img_predict, caption=predicted_emotion, use_column_width=True,channels="BGR")

        progress_bar.progress((i + 1) / num_files, text="Operation in progress. Please wait.")
        time.sleep(1)
        i = i + 1
    if i == num_files and i != 0:
        progress_bar.progress(i / num_files, text="Finished")
        end_time = time.time()
        # 计算预测所用的时间，并转换为毫秒
        prediction_time = int((end_time - start_time) * 1000)
        st.success("Success, it took "+ str(prediction_time) + " milliseconds")
        for i in range(0, len(img_show), 2):
            row_images = img_show[i:i + 2]
            row_emo = emo_show[i:i + 2]
            col1, col2= st.columns(2)
            with col1:
                if len(row_images) > 0:
                    st.image(row_images[0], caption=emo_show[0], use_column_width=True, channels="BGR")
            with col2:
                if len(row_images) > 1:
                    st.image(row_images[1], caption=emo_show[1], use_column_width=True, channels="BGR")


elif choice == "Webcam Face Detection":
    st.header("Webcam Live")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=Faceemotion)

elif choice == "Image Cropper":
    st.header("Image Cropper")
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
    aspect_dict = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "2:3": (2, 3),
        "Free": None
    }
    aspect_ratio = aspect_dict[aspect_choice]

    if img_file:
        img = Image.open(img_file)
        if not realtime_update:
            st.write("Double click to save crop")
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,aspect_ratio=aspect_ratio)

        _ = cropped_img.thumbnail((150, 150))
        col1, col2 = st.columns(2)
        with col1:
            st.write("Preview")
            st.image(cropped_img)
        # image = np.array(bytearray(cropped_img), dtype=np.uint8)

        image = np.array(cropped_img)

        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img_cropper = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_resize = np.expand_dims(img_cropper, axis=-1)
        faces = classifier.detectMultiScale(image=img_resize, scaleFactor=1.3, minNeighbors=5)
        j=0
        for (x, y, w, h) in faces:

            roi_gray = img_resize[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.expand_dims(roi_gray, axis=0)
            img_pixels = img_pixels / 255.0

            predictions = model.predict(img_pixels)
            max_index = int(np.argmax(predictions))
            predicted_emotion = emotions[max_index]
            result = str(predicted_emotion) + " " + str(np.round(np.max(predictions[0]), 2))
            cv2.rectangle(image, (x, y), (x + w, y + h), color_RGB[max_index], 2)
            img_predict = cv2.putText(image, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_RGB[max_index], 2)
            j = j + 1
            if j == len(faces):
                with col2:
                    st.write("Expression")
                    st.image(img_predict, caption=result, width=150,channels="RGB")

elif choice == "User Manual":
    st.markdown("## Facial Expression Recognition Parameter Adjustment Guide")
    st.markdown("To enhance your experience on our facial expression recognition website, you are free to adjust the "
                "following two key parameters according to your needs:")
    st.markdown("### scaleFactor")
    st.markdown("The `scaleFactor` determines the scale reduction in image size during the detection process. "
                "Adjusting this parameter helps balance detection speed and accuracy:")
    st.markdown("- **Smaller values**: Make the detection process more thorough, capable of detecting smaller "
                "expressions, but will increase the time required for detection.")
    st.markdown("- **Larger values**: Speed up the detection process, but might miss some small or blurry expressions.")
    st.markdown("The recommended default value is `1.3`. You can start with this value and adjust gradually to find "
                "the setting that best suits your needs.")
    st.markdown("### minNeighbors")
    st.markdown("The `minNeighbors` parameter specifies the number of neighboring rectangles each candidate rectangle "
                "should have (i.e., the number of other expressions around a detected expression), affecting the "
                "quality of the detected expressions:")
    st.markdown("- **Higher values**: Only expressions surrounded by a larger number of other expressions are "
                "detected. This reduces false detections but might miss some isolated expressions.")
    st.markdown("- **Lower values**: Makes it easier to detect isolated expressions, even if there are no other "
                "expressions around. This might increase the number of false detections.")
    st.markdown("The recommended default value is `5`. Depending on your preference for accuracy and detection rate, "
                "you can adjust this parameter.")





