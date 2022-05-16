import os
import json
import time
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import random
import numpy as np
import imutils
import os
import cv2
from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__, static_folder='web', template_folder='web')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
counter = 0
inception_model_1 = load_model('resnet-inception-v2.h5')

def image_resizer(image):
    width = 224
    height = 224

    (img_height, img_width) = image.shape[:2]
    d_width = 0
    d_height = 0

    if img_width < img_height:
        image = imutils.resize(image, width=width, inter=cv2.INTER_AREA)
        d_height = int((image.shape[0] - height) / 2.0)
    else:
        image = imutils.resize(image, height=height, inter=cv2.INTER_AREA)
        d_width = int((image.shape[1] - width) / 2.0)

    (res_height, res_width) = image.shape[:2]
    image = image[d_height:res_height - d_height, d_width:res_width - d_width]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_to_array(image):
    return img_to_array(image)

def image_loader(file_path_name):
    data = []
    image = cv2.imread(file_path_name)
    image = preprocess(image)
    data.append(image)
    return (np.array(data))

def preprocess(image):
    image = image_resizer(image)
    image = image_to_array(image)
    return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/check-contamination", methods=['POST'])
def contamination_checker():
    dict_labels = {
        0: "CardBoard",
        1: "Metal",
        2: "Paper",
        3: "Plastic Bags",
        4: "Plastic Bottles",
        5: "Plastic Containers",
        6: "Takeaway Cups"
    }

    response_msg = ""
    contaminator = 0
    response_status = 400
    try:
        file_data = request.files['file']
        # print(file_data)
        if file_data.filename == '':
            response_msg = {"error": "Invalid file selected"}
            return Response(json.dumps(response_msg), status=response_status, mimetype='application/json')
        if file_data and allowed_file(file_data.filename):
            filename = secure_filename(file_data.filename)
            file_data.save(os.path.join("./web", filename))
            global inception_model_1
            print("./web/"+filename)

            inf_data = image_loader("./web/"+filename)
            print(inf_data.shape)
            inf_data = inf_data.astype("float") / 255.0
            pred_1 = inception_model_1.predict(inf_data, batch_size=1)
            index = pred_1.argmax(axis=1)[0]
            category = dict_labels.get(index)
            conf = (np.amax(pred_1))* 100

            if index == 3 or index == 6:
                contaminator = "yes"
            else:
                contaminator = "no"

            response_msg = {"prediction": category, "contaminator": contaminator, "confidence": str(format(conf, '.2f'))}
            response_status = 200
            return Response(json.dumps(response_msg), status=response_status, mimetype='application/json')
    except Exception as err:
        err_msg = str(err)
        err_response = {"error": err_msg}
        print(err_msg)
        return Response(json.dumps(err_response), status=400, mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))