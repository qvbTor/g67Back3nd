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
        measurements_file = r"C:\Users\PC\Desktop\ sdsds\g67Backend-main1\normalized_measurements.txt"
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
        realheightG = float(data['height'])
        filename = data.get('filename', 'uploaded_image.jpg')

        # Remove the prefix from the Base64 string if it exists
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        # Decode the Base64 string
        image_data = base64.b64decode(base64_image)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(image_data)

        print(f"Image saved successfully to {filepath}")

        normalized_measurements = get_measurements_from_user(realheightG)
        return jsonify({
            'message': 'File uploaded successfully',
            'normalized_measurements': normalized_measurements
        }), 200
    except Exception as e:
        print(f"Error in /upload: {e}")
        return jsonify({'error': str(e)}), 500





import math
import mediapipe as mp
import cv2

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle(p1, p2):
    """Calculate the angle between two points with respect to the horizontal."""
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return math.degrees(math.atan2(delta_y, delta_x))

def process_image_and_scale(image_path, real_height):
    """Process the image to extract pose keypoints using MediaPipe,
    and scale the keypoints based on the provided real height.
    """

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose.Pose()

    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect keypoints
    results = mp_pose.process(img_rgb)

    if results.pose_landmarks is None:
        print("No pose landmarks detected.")
        return None

    # Extract keypoints from results
    keypoints = {}
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        keypoints[idx] = (landmark.x, landmark.y)  # Store normalized (x, y) values

    # Calculate the height (vertical distance between the ankle and the top of the head)
    ankle = keypoints[27]  # Left Ankle (for example)
    head = keypoints[0]  # Nose (head top approximation)

    # Calculate the height in the image (distance between ankle and head)
    image_height = calculate_distance(head, ankle)

    # Now scale the keypoints using the real height provided by the user
    if image_height == 0:  # Avoid division by zero
        print("Error: Invalid height in keypoints.")
        return None

    ratio = real_height / image_height  # Calculate scale ratio

    # Scale the keypoints based on the ratio
    scaled_keypoints = {}
    for i in keypoints:
        # Scale the (x, y) positions and store them in scaled_keypoints
        scaled_keypoints[i] = (ratio * keypoints[i][0], ratio * keypoints[i][1])

    # Calculate measurements
    measurements = calculate_body_measurements(scaled_keypoints)

    return measurements

def calculate_body_measurements(keypoints):
    """Calculate body measurements based on pose keypoints."""

    # Chest Circumference (approximation using shoulder width and rib cage expansion)
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    chest_circumference = shoulder_width * 2  # Approximate

    # Waist Circumference (approximation using hips)
    left_hip = keypoints[23]
    right_hip = keypoints[24]
    waist_circumference = calculate_distance(left_hip, right_hip) * 2  # Approximate

    # Hip Circumference (around the widest point of hips)
    hip_circumference = waist_circumference  # Similar to waist approximation in pose

    # Inseam Length (distance from the hip to the ankle)
    left_ankle = keypoints[27]
    right_ankle = keypoints[28]
    left_leg_length = calculate_distance(left_hip, left_ankle)
    right_leg_length = calculate_distance(right_hip, right_ankle)
    inseam_length = (left_leg_length + right_leg_length) / 2

    # Arm Length (shoulder to wrist)
    left_wrist = keypoints[15]
    right_wrist = keypoints[16]
    left_arm_length = calculate_distance(left_shoulder, left_wrist)
    right_arm_length = calculate_distance(right_shoulder, right_wrist)

    # Shoulder Length (shoulder width)
    shoulder_length = calculate_distance(left_shoulder, right_shoulder)

    # Thigh Circumference (approximation using distance between hips and knees)
    left_knee = keypoints[25]  # Left Knee
    right_knee = keypoints[26]  # Right Knee
    left_thigh_circumference = calculate_distance(left_hip, left_knee) * 2  # Approximate
    right_thigh_circumference = calculate_distance(right_hip, right_knee) * 2  # Approximate

    # Neck Circumference (approximation based on neck length and head width)
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    neck_length = calculate_distance(left_shoulder, right_shoulder) * 0.25  # Approximate
    neck_circumference = neck_length * 3.14  # Using the neck length to estimate circumference

    reco = left_arm_length*1.1
    reco1 = chest_circumference*1.1
    measurements = {
        "A. Chest Circumference(cm)": chest_circumference,
        "B. Waist Circumference(cm)": waist_circumference,
        "C. Hip Circumference(cm)": hip_circumference,
        "D. Inseam Length(cm)": inseam_length,
        "E. Left Arm Length(cm)": left_arm_length,
        "F. Right Arm Length(cm)": right_arm_length,
        "G. Shoulder Length(cm)": shoulder_length,
        "H. Left Thigh Circumference(cm)": left_thigh_circumference,
        "I. Right Thigh Circumference(cm)": right_thigh_circumference,
        "J. Neck Circumference(cm)": neck_circumference,
        "K. Recommended width in cm: ": reco,
        "L. Recommended length in cm: ": reco1
    }

    return measurements

# Function to accept user input and process the image
def get_measurements_from_user(real_height):
    """Accept user input for image and height, and generate measurements."""
    image_path = r"C:\Users\PC\Desktop\sdsds\g67Backend-main1\uploads\uploaded_image.jpg"


    # Process the image and generate measurements
    measurements = process_image_and_scale(image_path, real_height)
    return measurements







if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


