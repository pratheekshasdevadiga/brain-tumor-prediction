import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import os
import joblib

app = Flask(__name__)
model = joblib.load('model/svm_model.pkl')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

classes = {0: 'No Tumor', 1: 'Positive Tumor'}

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image = cv2.imread(filepath, 0)  # Read image in grayscale
            image = cv2.resize(image, (200, 200))  # Resize image to 200x200
            image = image.reshape(1, -1) / 255  # Flatten and normalize image
            prediction = model.predict(image)[0]
            image_url = url_for('static', filename='uploads/' + file.filename)
    return render_template('index.html', prediction=classes[prediction] if prediction is not None else None, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
