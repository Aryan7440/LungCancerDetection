import os
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import logging

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
mobilenet_model = load_model('models/mobilenetv2_model.h5')

resnet_model = load_model('models/Resnet_model.h5')
efficientnet_model = load_model('models/EfficientNetB0_model.h5')
custom_model = load_model('models/custom_cnn_model.h5')

CLASS_NAMES = ['Bengin case', 'Malignant case', 'Normal case']  
def preprocess_custom_img_CNN(img_path):
    img = image.load_img(img_path, target_size=(512, 512))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(227, 227))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img_array = preprocess_img(filepath)
    img_array_custom = preprocess_custom_img_CNN(filepath)
    
    logging.basicConfig(level=logging.INFO)
    
    mobilenet_score = mobilenet_model.predict(img_array)
    resnet_score = resnet_model.predict(img_array)
    efficientnet_score = efficientnet_model.predict(img_array)
    custom_cnn_score = custom_model.predict(img_array_custom)
    
    logging.info(f"MobileNetV2 Score: {mobilenet_score}")
    logging.info(f"ResNet50 Score: {resnet_score}")
    logging.info(f"EfficientNetB0 Score: {efficientnet_score}")
    logging.info(f"Custom CNN Score: {custom_cnn_score}")

    preds = {
        "MobileNetV2": {
            "class": CLASS_NAMES[np.argmax(mobilenet_score)],
            "confidence": round(np.max(mobilenet_score) * 100,2)
        },
        "ResNet50": {
            "class": CLASS_NAMES[np.argmax(resnet_score)],
            # "confidence": round(np.max(resnet_score) * 100,2)
            "confidence": "Low confidence"
        },
        "EfficientNetB0": {
            "class": CLASS_NAMES[np.argmax(efficientnet_score)],
            # "confidence": round(np.max(efficientnet_score) * 100,2)
             "confidence": "Low confidence"
        },
        "CustomCNNModel": {
            "class": CLASS_NAMES[np.argmax(custom_cnn_score)],
            "confidence": round(np.max(custom_cnn_score) * 100,2)
        }
    }

    return render_template('index.html', image=file.filename, predictions=preds)

if __name__ == '__main__':
    app.run(debug=True)
