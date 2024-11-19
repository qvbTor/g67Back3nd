import math
import os

import mediapipe as mp
import cv2
import numpy as np


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle(p1, p2):
    """Calculate the angle between two points with respect to the horizontal."""
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return math.degrees(math.atan2(delta_y, delta_x))

import os
import cv2
import numpy as np
import mediapipe as mp

def process_image_and_scale(image_path, real_height):
    """Process the image to extract pose keypoints using MediaPipe,
    and scale the keypoints based on the provided real height.
    """

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose.Pose()

    # Debug: Check if the image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load the image. Ensure {image_path} is a valid image file.")

    # Debug: Print the image shape and type
    print(f"Image loaded successfully. Shape: {img.shape}, Type: {type(img)}")

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Debug: Validate image dtype
    print(f"Image dtype before conversion: {img_rgb.dtype}")
    if img_rgb.dtype != np.uint8:
        print("Converting image to uint8...")
        img_rgb = img_rgb.astype(np.uint8)

    # Validate image dimensions and data type
    if not isinstance(img_rgb, np.ndarray) or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Invalid image format. Image must be a numpy.ndarray with 3 channels.")

    # Debug: Validate image properties
    print(f"Image dtype after conversion: {img_rgb.dtype}, Shape: {img_rgb.shape}")

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
    try:
        ankle = keypoints[27]  # Left Ankle (for example)
        head = keypoints[0]  # Nose (head top approximation)
        image_height = calculate_distance(head, ankle)
    except KeyError:
        print("Missing key landmarks for measurement calculation.")
        return None

    if image_height == 0:  # Avoid division by zero
        print("Error: Invalid height in keypoints.")
        return None

    # Calculate the scale ratio
    ratio = real_height / image_height

    # Scale the keypoints based on the ratio
    scaled_keypoints = {i: (ratio * keypoints[i][0], ratio * keypoints[i][1]) for i in keypoints}

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

    measurements = {
        "Chest Circumference": chest_circumference,
        "Waist Circumference": waist_circumference,
        "Hip Circumference": hip_circumference,
        "Inseam Length": inseam_length,
        "Left Arm Length": left_arm_length,
        "Right Arm Length": right_arm_length,
        "Shoulder Length": shoulder_length,
        "Left Thigh Circumference": left_thigh_circumference,
        "Right Thigh Circumference": right_thigh_circumference,
        "Neck Circumference": neck_circumference
    }

    return measurements

# Function to accept user input and process the image
def get_measurements_from_user():
    """Accept user input for image and height, and generate measurements."""

    # Accept user image and height as input
    image_path = r'C:\Users\migzuu\PycharmProjects\pythonProject5\uploads\462550064_1112784256864921_4481066422507567856_n.jpg'
    real_height = 170

    # Process the image and generate measurements
    measurements = process_image_and_scale(image_path, real_height)

    # Print the measurements
    if measurements:
        print("\nCalculated Body Measurements:")
        for name, value in measurements.items():
            print(f"{name}: {value:.2f} meters")
    else:
        print("Error processing the image or calculating the measurements.")

# Run the program
get_measurements_from_user()

