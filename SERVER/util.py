import json
import base64
import cv2
from wavelet import wavelet
import numpy as np
import joblib
import pickle
num_to_class_name = None
class_name_to_num = None

model = None


def classify_image(base_64, file_path=None):
    image = cropped_image(file_path, base_64)
    # print(image)
    result = []
    for img in image:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = wavelet(img, 5, 'db1')
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(
            32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({'class': get_name_from_num(model.predict(final)[0]), 'class_prob': np.round(
            model.predict_proba(final), 2), 'class_dictionary': class_name_to_num})
    return result


def get_name_from_num(number):
    return num_to_class_name[number]


def load_saved_artifacts():
    global class_name_to_num
    global num_to_class_name
    with open('SERVER/artifacts/name_dict.json', 'r') as f:
        class_name_to_num = json.load(f)
        num_to_class_name = {v: k for k, v in class_name_to_num.items()}

    global model
    if model is None:
        with open('SERVER/artifacts/Footballer_predictor.pickle', 'rb') as f:
            model = pickle.load(f)
    print("loading saved artifacts...done")


def get_cv2_img_from_b64string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cropped_image(image_path, image_b64_data):
    face_classifier = cv2.CascadeClassifier(
        'SERVER/OpenCV/HaarCascade/haarcascade_frontalface_default.xml')
    eyes_classifier = cv2.CascadeClassifier(
        'SERVER/OpenCV/HaarCascade/haarcascade_eye.xml')
    if image_path:
        image = cv2.imread(image_path)
    else:
        image = get_cv2_img_from_b64string(image_b64_data)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eyes_classifier.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def get_base_64():
    with open('SERVER/b64.txt') as f:
        return f.read()


if (__name__ == "__main__"):
    load_saved_artifacts()
    print(classify_image(get_base_64(), None))
    # print(get_base_64())
    # print(cropped_image(None, get_base_64()))
    # print(classify_image(None, 'SERVER/testing/68290-1669394812.jpg'))
