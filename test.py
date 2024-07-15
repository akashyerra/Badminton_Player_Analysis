import tensorflow as tf
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

EDGES = {
    (0, 1): (255, 182, 193),  # Nose to Left eye (Light Pink)
    (0, 2): (255, 182, 193),  # Nose to Right eye (Light Pink)
    (1, 3): (255, 105, 180),  # Left eye to Left ear (Hot Pink)
    (2, 4): (255, 105, 180),  # Right eye to Right ear (Hot Pink)
    (0, 5): (135, 206, 250),  # Nose to Left shoulder (Light Sky Blue)
    (0, 6): (135, 206, 250),  # Nose to Right shoulder (Light Sky Blue)
    (5, 7): (255, 165, 0),    # Left shoulder to Left elbow (Orange)
    (7, 9): (255, 69, 0),     # Left elbow to Left wrist (Red Orange)
    (6, 8): (255, 165, 0),    # Right shoulder to Right elbow (Orange)
    (8, 10): (255, 69, 0),    # Right elbow to Right wrist (Red Orange)
    (5, 6): (173, 216, 230),  # Left shoulder to Right shoulder (Light Blue)
    (5, 11): (144, 238, 144), # Left shoulder to Left hip (Light Green)
    (6, 12): (144, 238, 144), # Right shoulder to Right hip (Light Green)
    (11, 13): (0, 128, 0),    # Left hip to Left knee (Green)
    (13, 15): (34, 139, 34),  # Left knee to Left ankle (Forest Green)
    (12, 14): (0, 128, 0),    # Right hip to Right knee (Green)
    (14, 16): (34, 139, 34)   # Right knee to Right ankle (Forest Green)
}

ANGLE_EDGES = {
    (5, 7, 9): 'Left Shoulder - Left Elbow - Left Wrist',
    (6, 8, 10): 'Right Shoulder - Right Elbow - Right Wrist',
    (11, 13, 15): 'Left Hip - Left Knee - Left Ankle',
    (12, 14, 16): 'Right Hip - Right Knee - Right Ankle'
}

# Define the desired dimensions for resizing
target_width = 256
target_height = 256

def preprocess_image(image_path, target_width, target_height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (target_width, target_height))

    img = tf.image.resize_with_pad(np.expand_dims(resized_image, axis=0), target_width, target_height)
    input_image = tf.cast(img, dtype=tf.float32)
    
    return input_image, resized_image


# Load the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path="D:\\CODING\ML\\A10_MINI_PROJECT\\PoseEstimation\\t_3.tflite")
interpreter.allocate_tensors()


def get_keypoints(input_image, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for edge, edge_color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), edge_color, 4)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 0), -1)

def calculate_angle(p1, p2, p3):
    a = np.array(p1[:2])
    b = np.array(p2[:2])
    c = np.array(p3[:2])
    
    ab = a - b
    cb = c - b
    
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# Extract x, y coordinates of keypoints
def get_keypoints_xy(keypoints_with_scores, target_height, target_width):
    return np.squeeze(np.multiply(keypoints_with_scores[..., :2], [target_height, target_width]))


# print(keypoints_xy1)
# print(keypoints_xy2)

def get_angle_dict(keypoints, edges):
    angles = {}
    for edge in edges:
        if len(edge) == 3:
            p1, p2, p3 = edge
            angles[edge] = calculate_angle(keypoints[p1], keypoints[p2], keypoints[p3])
    return angles


def plot_pose_with_angles(keypoints1, keypoints2, edges):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    max_x = max(np.max(keypoints1[:, 1]), np.max(keypoints2[:, 1]))
    max_y = max(np.max(keypoints1[:, 0]), np.max(keypoints2[:, 0]))
    max_value = max(max_x, max_y)
    
    ax1.set_xlim(0, max_value)
    ax1.set_ylim(0, max_value)
    ax2.set_xlim(0, max_value)
    ax2.set_ylim(0, max_value)
    
    ax1.set_xticks(np.arange(0, max_value, 16))
    ax1.set_yticks(np.arange(0, max_value, 16))
    ax1.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax2.set_xticks(np.arange(0, max_value, 16))
    ax2.set_yticks(np.arange(0, max_value, 16))
    ax2.grid(True, linestyle='--', color='gray', alpha=0.5)
    
    ax1.plot([], [], 'b-', linewidth=2, label='BEGINNER')
    
    ax2.plot([], [], 'r-', linewidth=2, label='PROFESSIONAL')
    

    for edge in EDGES:
        p1, p2 = edge
        ax1.plot([keypoints1[p1][1], keypoints1[p2][1]], [keypoints1[p1][0], keypoints1[p2][0]], 'b-', linewidth=2)
        ax2.plot([keypoints2[p1][1], keypoints2[p2][1]], [keypoints2[p1][0], keypoints2[p2][0]], 'r-', linewidth=2)

    angle_dict1 = get_angle_dict(keypoints1, edges)
    for edge, angle in angle_dict1.items():
        p1, p2, p3 = edge
        ax1.text(keypoints1[p2][1], keypoints1[p2][0], f'{angle:.1f}', fontsize=12, color='red')
    
    angle_dict2 = get_angle_dict(keypoints2, edges)
    for edge, angle in angle_dict2.items():
        p1, p2, p3 = edge
        ax2.text(keypoints2[p2][1], keypoints2[p2][0], f'{angle:.1f}', fontsize=12, color='blue')
    
    ax1.legend()
    ax2.legend()
    
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    
    ax2.set_title('PROFESSIONAL')
    ax1.set_title('BEGINNER')
    
    plt.tight_layout()
    fig = plt.gcf()
    return fig, angle_dict1, angle_dict2

# Draw keypoints and connections on the original images
def draw_pose_on_image(image, keypoints_with_scores, edges, confidence_threshold):
    image_copy = image.copy()
    draw_connections(image_copy, keypoints_with_scores, edges, confidence_threshold)
    draw_keypoints(image_copy, keypoints_with_scores, confidence_threshold)
    return image_copy



# Plot pose with angles

def angles_plot(keypoints_xy1, keypoints_xy2):
    fig, angle_dict1, angle_dict2 = plot_pose_with_angles(keypoints_xy1, keypoints_xy2, list(ANGLE_EDGES.keys()))

    comparision_statements = ["\nComparison of angles between poses:"]
    # Compare angles
    print("\nComparison of angles between poses:")
    for edge, description in ANGLE_EDGES.items():
        angle1 = angle_dict1.get(edge, None)
        angle2 = angle_dict2.get(edge, None)
        if angle1 and angle2:
            statement = (
                f'{description}:\n\n'
                f'  BEGINNER: {angle1:.1f} degrees\n\n'
                f'  PROFESSIONAL: {angle2:.1f} degrees\n\n'
                f'  Difference: {(angle1 - angle2):.1f} degrees\n'
            )
        comparision_statements.append(statement)

    return fig, comparision_statements


def poseEstimation(image1, image2, keypoints_with_scores1, keypoints_with_scores2):
    image1_with_pose = draw_pose_on_image(image1, keypoints_with_scores1, EDGES, 0.4)
    image2_with_pose = draw_pose_on_image(image2, keypoints_with_scores2, EDGES, 0.4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    ax1.imshow(cv2.cvtColor(image1_with_pose, cv2.COLOR_BGR2RGB))
    ax1.set_title('BEGINNER')

    ax2.imshow(cv2.cvtColor(image2_with_pose, cv2.COLOR_BGR2RGB))
    ax2.set_title('PROFESSIONAL')

    fig = plt.gcf()
    return fig

def main (img1, img2):
    input_image1, image1 = preprocess_image(img1, target_width, target_height)
    input_image2, image2 = preprocess_image(img2, target_width, target_height)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    keypoints_with_scores1 = get_keypoints(input_image1, input_details, output_details)
    keypoints_with_scores2 = get_keypoints(input_image2, input_details, output_details)

    keypoints_xy1 = get_keypoints_xy(keypoints_with_scores1, target_height, target_width)
    keypoints_xy2 = get_keypoints_xy(keypoints_with_scores2, target_height, target_width)

    pose_graph = poseEstimation(image1, image2, keypoints_with_scores1, keypoints_with_scores2)

    angles_graph, comparision_statements = angles_plot(keypoints_xy1, keypoints_xy2)
    return pose_graph, angles_graph, comparision_statements


# image1 = input("Enter the address of image1: ")
# image2 = input("Enter the address of image2: ")
# main(image1, image_2)