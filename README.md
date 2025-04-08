# 🫁 Lung Cancer Detection using Deep Learning

This project is a Flask-based web application that uses multiple deep learning models to classify lung cancer from chest scan images. It leverages transfer learning with pre-trained architectures like **MobileNetV2**, **ResNet50**, **EfficientNetB0**, and a custom **CNN model** trained from scratch.

---

## 🚀 Features

- Upload and classify lung cancer images
- Uses multiple models to give comparative predictions
- Displays predicted class and confidence for each model
- Web interface built with Flask
- Models trained on 3 classes: `Normal`, `Benign`, `Malignant`

---

## 🧠 Models Used

1. **MobileNetV2** (Transfer Learning)
2. **ResNet50** (Transfer Learning)
3. **EfficientNetB0** (Transfer Learning with fine-tuning)
4. **Custom CNN** (Built and trained from scratch)

---

---

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection
```

### 2. Set up virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open your browser and go to <http://127.0.0.1:5000>

## 📊 Model Training (Summary)

All models were trained using **TensorFlow/Keras**. The training setup included:

### 🔧 Input Sizes

- **MobileNetV2 / ResNet50**: `227x227`
- **EfficientNetB0**: `224x224`
- **Custom CNN**: `512x512`

### ⚙️ Configuration

- **Optimizer**: `Adam`
- **Loss Function**: `SparseCategoricalCrossentropy`
- **Metrics**: `Accuracy`

> 🔍 **Note:** EfficientNetB0 includes a **fine-tuning phase** with top layers unfrozen.

---

## 🖼️ Example Prediction Output

- **MobileNetV2**: `Benign (98.3%)`
- **ResNet50**: `Malignant (60.6%)`
- **EfficientNetB0**: `Benign (58.4%)`
- **Custom CNN**: `Normal (98.5%)`

---

## 🧾 License

MIT License – free to use and modify.

## 🙌 Credits

This project was developed by **Aryan Shukla** as part of a lung cancer detection initiative using deep learning.

### Tools & Frameworks Used

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Flask](https://flask.palletsprojects.com/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/) *(for visualization, if used)*

### Pretrained Models

- MobileNetV2 (ImageNet weights)
- ResNet50 (ImageNet weights)
- EfficientNetB0 (ImageNet weights)

### Dataset

Lung cancer image dataset used for training and validation was obtained from kaggle NQTTH.  

---

> Special thanks to the open-source community for their amazing tools and resources.
