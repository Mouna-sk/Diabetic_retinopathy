import os
from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)


model = load_model(r'diabetic_retinopathy_model2.h5')


class_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file uploaded", 400
 
    img_path = os.path.join('static', 'uploads', file.filename)
    file.save(img_path)

   
    img = Image.open(file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    
    probabilities = model.predict(img_array)[0]
    predicted_label = class_labels[np.argmax(probabilities)]

    return render_template('result.html', predicted_label=predicted_label, probabilities=probabilities, img_path=img_path, class_labels=class_labels)

if __name__ == '__main__':
    app.run(port=5001,debug=True)