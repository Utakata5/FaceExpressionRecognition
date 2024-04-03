import cv2
import numpy as np
import tensorflow as tf



# Parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('image', help='path to input image file')
# args = vars(ap.parse_args())

# Load model from JSON file
# json_file = open('../Saved-Models/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# Load weights and them to model
# model.load_weights('../Saved-Models/model4000.h5')

# loaded_model = tf.keras.models.load_model('../Saved-Models/model4378.h5')
print("load model")
model=tf.keras.models.load_model('../Saved-Models/model4829.h5')
print(model)
print("load classifier")
classifier = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

img = cv2.imread('../testpic/outu.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
img_resize = np.expand_dims(img, axis=-1)
print(img_resize)
X_batch = np.expand_dims(img_resize, axis=0)
print("face")
faces_detected = classifier.detectMultiScale(img, 1.18, 5)
#
# print(img.shape)
# img_resize = np.expand_dims(img, axis=-1)
# print("after")
# print(img_resize.shape)
# print(img_resize)

# data = pd.read_csv('../fer2013.csv')
# X = np.fromstring(data['pixels'][1], dtype=int, sep=' ').reshape((48, 48, 1))
# X = X / 255.0
# X_batch = np.expand_dims(img_resize, axis=0)
# print(X)
#
# predictions = model.predict(X_batch)
# print(predictions)
# max_index = int(np.argmax(predictions))
# print(max_index)
#
# emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
# predicted_emotion = emotions[max_index]
# print(predicted_emotion)

for (x, y, w, h) in faces_detected:
    print("start")
    cv2.rectangle(img_resize, (x, y), (x + w, y + h), (0, 0, 0), 2)
    roi_gray = img_resize[y:y + w, x:x + h]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    # # img_pixels = image.img_to_array(roi_gray)
    # img_pixels = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
    # print(img_pixels.shape)
    print(roi_gray.shape)
    img_pixels = np.expand_dims(roi_gray, axis=0)
    print(img_pixels.shape)
    img_pixels = img_pixels / 255.0
    predictions = model.predict(img_pixels)
    max_index = int(np.argmax(predictions))
    print("Result")
    print(predictions)
    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    predicted_emotion = emotions[max_index]
    print(predicted_emotion)
    print(np.max(predictions[0]))
    print("12")

    cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

resized_img = cv2.resize(img, (1024, 768))
cv2.imshow('Facial Emotion Recognition', resized_img)

cv2.waitKey(0)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
