from flask import Flask, render_template, request, jsonify
import os
import base64
import pickle
import numpy as np
import cv2

app = Flask(__name__)

# Load your model (ensure model.pkl is in your project)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create uploads folder if needed
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Label mapping for your model
label_map = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

def preprocess_frame(image):
    """
    Preprocess the OpenCV image (BGR) for your model.
    Adjust this to match your model's input size and channels.
    Here, we convert to grayscale, resize to 48x48, and normalize.
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48
    gray = cv2.resize(gray, (48, 48))
    # Normalize [0,1]
    gray = gray.astype('float32') / 255.0
    # Reshape to (1, 48, 48, 1)
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    return gray

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Handle file-based upload
        if 'file' not in request.files:
            result = "No file part in the request."
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No file selected."
            else:
                # Save the file to uploads folder
                filename = file.filename  # You might want to secure the filename using secure_filename()
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Read the image using OpenCV
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if image is None:
                    result = "Could not read the uploaded image."
                else:
                    # Preprocess and predict
                    processed = preprocess_frame(image)
                    prediction = model.predict(processed)[0]
                    pred_index = np.argmax(prediction)
                    pred_label = label_map.get(pred_index, "Unknown")
                    result = f"Prediction: {pred_label}"
    return render_template('index.html', result=result)

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data['image']
    # Split header if present
    if ',' in image_data:
        header, encoded = image_data.split(',', 1)
    else:
        encoded = image_data

    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Decoding image failed'}), 400

    processed = preprocess_frame(frame)
    prediction = model.predict(processed)[0]
    pred_index = np.argmax(prediction)
    pred_label = label_map.get(pred_index, "Unknown")

    return jsonify({'prediction': pred_label})

if __name__ == '__main__':
    app.run(debug=True)
