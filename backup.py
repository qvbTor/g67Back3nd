import base64
import os
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import subprocess
import math
import mediapipe as mp
import cv2
import numpy as np
from testt import execute_with_scaling
import json


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for testing


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global realheightG






@app.route('/model')
def get_model():
    return send_file('Adjusted_Mannequin.obj', as_attachment=False)

@app.route('/generate', methods=['POST'])
def generate_model():
    try:
        normalized_measurements = execute_with_scaling(realheightG)
        print("WOW: ", normalized_measurements)
        print("Received normalized measurements:", normalized_measurements)
        # Write normalized_measurements to a text file
        measurements_file = r"C:\Users\PC\Desktop\sdsds\g67Backend-main1\normalized_measurements.txt"
        try:
            with open(measurements_file, 'w') as file:
                json.dump(normalized_measurements, file)
        except:
            print("ERROROROROROROROROROROROR")
        # Example Blender subprocess call (customize paths as needed)
        blender_command = [
            "blender",
            r"C:\Users\PC\Desktop\testBlend\test.blend",
            "--background",
            "--python",
            "randomize_bones.py"
        ]

        print("Running Blender command:", blender_command)
        subprocess.run(blender_command, check=True)
        return jsonify({"message": "Model generation successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    global realheightG
    try:
        data = request.get_json()
        if 'file' not in data or 'height' not in data:
            return jsonify({'error': 'Missing file or height in request'}), 400

        base64_image = data['file']
        realheightG = float(data['height'])  # User-provided height
        print("WOWWWWWWWWWWWWWWWWWWWWWWWWWW",realheightG)
        filename = data.get('filename', 'uploaded_image.jpg')

        image_data = base64.b64decode(base64_image)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_path = "uploads/uploaded_image.jpg"
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return jsonify({'message': 'File uploaded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


