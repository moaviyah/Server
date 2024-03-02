from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from io import BytesIO
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model_inception.h5')
specified_classes = ['wood', 'plastic', 'glass', 'cement']

@app.route('/classify-image', methods=['POST'])
def classify_image():
    # Check if request contains image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load and preprocess the imageA
    img_file = request.files['image']
    img = Image.open(BytesIO(img_file.read()))
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Use the model to predict the class of the image
    preds = model.predict(x)

    # Map the predicted class index to the corresponding class label
    predicted_class_index = np.argmax(preds)
    predicted_class_label = 'others' if np.max(preds) < 0.5 else specified_classes[predicted_class_index]

    return jsonify({'predicted_class': predicted_class_label}), 200

if __name__ == '__main__':
    app.run(debug=True)
