import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import logging

# Configure logging to suppress TensorFlow warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = Flask(__name__)

# Global settings
detector_backend = "opencv"  # Using OpenCV for better compatibility

# Pre-load models to reduce initial delay
try:
    print(f"Loading models using {detector_backend} backend, this might take a moment...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_frame, actions=["emotion"], 
                    detector_backend=detector_backend, enforce_detection=False)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Could not pre-load models: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    try:
        # Get the image data from the request
        image_data = request.json.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Remove the data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Analyze the frame - only emotion detection
        results = DeepFace.analyze(frame, actions=["emotion"], 
                                  detector_backend=detector_backend, 
                                  enforce_detection=False, 
                                  silent=True)
        
        # Process results - removed age
        processed_results = []
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict) and "region" in result:
                    region = result.get("region", {})
                    processed_result = {
                        'region': {
                            'x': region.get('x', 0),
                            'y': region.get('y', 0),
                            'w': region.get('w', 0),
                            'h': region.get('h', 0)
                        },
                        'dominant_emotion': result.get('dominant_emotion', 'N/A')
                    }
                    processed_results.append(processed_result)
        
        return jsonify({'results': processed_results})
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Explicitly bind to localhost on port 8080
    print("Starting server on http://localhost:8080")
    app.run(host='localhost', port=8080, debug=True)
