import base64
import os
import math
import subprocess
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import mediapipe as mp
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global realheightG

# Helper Functions
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def process_image_and_scale(image_path, real_height):
    """Process the image to extract pose keypoints and scale them."""
    mp_pose = mp.solutions.pose.Pose()
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(img_rgb)

    if results.pose_landmarks is None:
        print("No pose landmarks detected.")
        return None

    keypoints = {idx: (lm.x, lm.y) for idx, lm in enumerate(results.pose_landmarks.landmark)}

    # Calculate the height (distance between the nose and left ankle)
    ankle = keypoints[27]  # Left Ankle
    head = keypoints[0]  # Nose (head approximation)
    image_height = calculate_distance(head, ankle)

    if image_height == 0:
        print("Error: Invalid height in keypoints.")
        return None

    # Scale the keypoints based on real height
    ratio = real_height / image_height
    scaled_keypoints = {i: (ratio * kp[0], ratio * kp[1]) for i, kp in keypoints.items()}

    # Calculate body measurements
    measurements = calculate_body_measurements(scaled_keypoints)
    return measurements

def calculate_body_measurements(keypoints):
    """Calculate body measurements based on scaled keypoints."""
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    left_hip = keypoints[23]
    right_hip = keypoints[24]
    left_ankle = keypoints[27]
    right_ankle = keypoints[28]
    left_wrist = keypoints[15]
    right_wrist = keypoints[16]

    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    waist_circumference = calculate_distance(left_hip, right_hip) * 2
    inseam_length = (calculate_distance(left_hip, left_ankle) + calculate_distance(right_hip, right_ankle)) / 2
    arm_length = (calculate_distance(left_shoulder, left_wrist) + calculate_distance(right_shoulder, right_wrist)) / 2

    measurements = {
        "shoulders control": shoulder_width,
        "torso control": waist_circumference,
        "legs control": inseam_length,
        "arms control": arm_length,
    }
    return measurements

def normalize_measurements(measurements, ranges):
    """Normalize measurements to the defined ranges."""
    normalized = {}
    for part, (min_val, max_val) in ranges.items():
        value = measurements.get(part, 0)
        normalized[part] = (value - min_val) / (max_val - min_val)
    return normalized

# Flask Endpoints
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and calculate measurements."""
    global realheightG
    try:
        data = request.get_json()
        if 'file' not in data or 'height' not in data:
            return jsonify({'error': 'Missing file or height in request'}), 400

        base64_image = data['file']
        realheightG = float(data['height'])
        filename = data.get('filename', 'uploaded_image.jpg')

        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(base64_image))

        # Process the image and calculate measurements
        measurements = process_image_and_scale(filepath, realheightG)
        if measurements is None:
            return jsonify({'error': 'No pose landmarks detected in the image.'}), 400

        # Normalize the measurements
        body_part_ranges = {
            "shoulders control": (0, 0.26),
            "torso control": (-0.04, 0.19),
            "legs control": (-0.035, 0.22),
            "arms control": (-0.04, 0.215),
        }
        normalized = normalize_measurements(measurements, body_part_ranges)

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'measurements': measurements,
            'normalized_measurements': normalized
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_model():
    """Generate 3D model based on normalized measurements."""
    try:
        data = request.get_json()
        if 'normalized_measurements' not in data:
            return jsonify({'error': 'Missing normalized measurements in request'}), 400

        normalized_measurements = data['normalized_measurements']
        print("Received normalized measurements:", normalized_measurements)
        # Write normalized_measurements to a text file
        measurements_file = r"C:\Users\PC\Desktop\sdsds\g67Backend-main1\normalized_measurements.txt"
        with open(measurements_file, 'w') as file:
            json.dump(normalized_measurements, file)


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
    except subprocess.CalledProcessError as e:
        print("Blender subprocess failed:", e)
        return jsonify({'error': f"Blender subprocess failed: {str(e)}"}), 500
    except Exception as e:
        print("Error in /generate:", e)
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

@app.route('/model', methods=['GET'])
def get_model():
    """Serve the generated model."""
    try:
        model_path = 'Adjusted_Mannequin.obj'
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        return send_file(model_path, as_attachment=False)
    except Exception as e:
        print("Error in /model:", e)
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
