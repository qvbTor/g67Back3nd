import base64
import os
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import subprocess
import math
import mediapipe as mp
import cv2
import numpy as np


body_part_ranges = {
            "shoulder angle": (0, 0.22),
            "shoulders control": (0, 0.26),
            "neck control": (0, 0.185),
            "breasts control": (0, 0.26),
            "breasts angle": (0, 0.11),
            "torso control": (0, 0.19),
            "hips control": (0, 0.21),
            "legs control": (0, 0.22),
            "belly control": (0, 0.22),
            "arms control": (0, 0.215),
        }

max_values = {
    "shoulder angle": 180,       # Degrees
    "shoulders control": 55,    # cm
    "neck control": 45,         # cm
    "breasts control": 120,     # cm
    "breasts angle": 90,        # Degrees
    "torso control": 110,       # cm
    "hips control": 130,        # cm
    "legs control": 120,        # cm
    "belly control": 110,       # cm
    "arms control": 70,         # cm
}



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
    print("Read")
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the image and detect keypoints
    print("IPAPASOK")
    results = mp_pose.process(img_rgb)
    print("NAPASOK")
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
    measurements = calculate_body_measurements(keypoints)

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

    print(chest_circumference)
    print(waist_circumference)
    print(hip_circumference)
    print(inseam_length)
    print(left_arm_length)
    print(shoulder_length)
    print(neck_circumference)

    measurements = {
        "breasts control": chest_circumference/max_values['breasts control'] * max(body_part_ranges['breasts control'])*50,  # Chest Circumference
        "torso control": waist_circumference/max_values['torso control']* max(body_part_ranges['torso control'])*130,  # Waist Circumference
        "hips control": hip_circumference/max_values['hips control']* max(body_part_ranges['hips control'])*130,  # Hip Circumference
        "legs control": inseam_length/max_values['legs control']* max(body_part_ranges['legs control'])*130,  # Inseam Length
        "arms control": left_arm_length/max_values['arms control']* max(body_part_ranges['arms control'])*130,  # Left Arm Length
        "shoulders control": shoulder_length/max_values['shoulders control']* max(body_part_ranges['shoulders control'])*130,  # Shoulder Length
        "neck control": neck_circumference/max_values['neck control']* max(body_part_ranges['neck control'])*130  # Neck Circumference
    }
    print(measurements)
    return measurements

def get_measurements_from_user(image_path, real_height):
    print("WOWWWOWOWOOW")
    measurements = process_image_and_scale(image_path, real_height)

    print(measurements)

    return measurements

def scale_measurements_with_ranges(measurements, ranges):
    """Strictly scale measurements using only the given ranges."""
    scaled = {}
    for part, (min_val, max_val) in ranges.items():
        value = measurements.get(part, 0)

        # Clamp the value to the range
        clamped_value = clamp(value, min_val, max_val)

        # Ensure normalized values are strictly within [0, 1]
        if max_val - min_val != 0:
            normalized_value = (clamped_value - min_val) / (max_val - min_val)
        else:
            normalized_value = 0

        # Log clamping and scaling details for debugging
        print(f"{part}: Original={value}, Clamped={clamped_value}, Normalized={normalized_value:.3f}")

        # Assign normalized value to the scaled dictionary
        scaled[part] = normalized_value

    return scaled

def execute_with_scaling(real_height):
    image_path = r"C:\Users\PC\Desktop\sdsds\g67Backend-main1\uploads\462550064_1112784256864921_4481066422507567856_n.jpg"

    # Get raw measurements from the user
    measurements = get_measurements_from_user(image_path, real_height)
    print("Raw Measurements:", measurements)



    return measurements


scaled_results = execute_with_scaling(170)
print("Scaled Measurements:", scaled_results)
