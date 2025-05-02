from flask import Flask, render_template, request, jsonify, Response
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import uuid
import time

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained emotion detection model
# You'll need to download or train this model separately
model_path = 'fer2013_cnn_model.h5'  # Replace with your model path
try:
    emotion_model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    emotion_model = None

# Emotion labels (should match your model output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Face detection using Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Generate unique filename
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image for emotion detection
        result = process_image(filepath)
        
        if result['success']:
            # Save processed image with face detection
            display_filename = f"processed_{filename}"
            display_filepath = os.path.join(app.config['UPLOAD_FOLDER'], display_filename)
            # cv2.imwrite(display_filepath, result['annotated_image'])
            
            return jsonify({
                'emotion': result['emotion'],
                'confidence': f"{result['confidence']:.2f}%",
                # 'display_image': display_filename
            })
        else:
            return jsonify({'error': result['error']})

def process_image(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not read image'}
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {'success': False, 'error': 'No faces detected in the image'}
        
        # Process the first detected face
        x, y, w, h = faces[0]
        
        # Draw rectangle around the face
        # annotated_image = image.copy()
        # cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess the face for the model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # Get emotion prediction
        if emotion_model is not None:
            predictions = emotion_model.predict(face_roi)[0]
            emotion_idx = np.argmax(predictions)
            emotion = emotion_labels[emotion_idx]
            confidence = predictions[emotion_idx] * 100
            
            # # Add emotion text to the image
            # cv2.putText(annotated_image, emotion, (x, y-10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': [float(p * 100) for p in predictions],
                # 'annotated_image': annotated_image
            }
        else:
            return {'success': False, 'error': 'Emotion model not loaded'}
    
    except Exception as e:
        return {'success': False, 'error': f'Error processing image: {str(e)}'}



if __name__ == '__main__':
    app.run(debug=True)