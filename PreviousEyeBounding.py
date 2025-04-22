import math

import cv2
import dlib
import numpy as np

# Define the fixed window size
WINDOW_SIZE = (200, 100)

# open the camera
def create_cam():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)
    # Exit program if we can't open the camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    return detector, predictor, cap

# using our existing model we place the necessary points to focus on our eyes
def facial_landmarks(predictor, face, landmarks, lB, uB):
    shape = predictor(face, landmarks)
    eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(lB, uB)], np.int32)
    eye_center = np.mean(eye_points, axis=0).astype(int)
    return eye_points, eye_center

# draw the eye points onto a person's face
def draw_landmarks(frame, eye_points):
    for (x, y) in eye_points:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

def eye_orientation(left_eye_center, right_eye_center):
    """
    Calculate the angle of rotation between the left and right eye centers,
    and compute the midpoint (eyes_center) between them. 
    """
    
    angle = math.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0, (left_eye_center[1] + right_eye_center[1]) / 2.0)
    return angle, eyes_center

def orient_eyes(frame, detector, eyes_center, angle):
    """
    Rotate the frame around the 'eyes_center' point by 'angle' degrees,
    then run face detection on the rotated gray frame to find the new face bounding rect.
    Returns the rotated color frame, the rotated gray frame, and the detection results.
    """
    
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    gray_rotated = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
    dets_rotated = detector(gray_rotated)
    return rotated_frame, gray_rotated, dets_rotated

def calculate_fixed_bounding_box(center, frame_shape, window_size):
    """
    Given a 'center' (x_center, y_center) and desired 'window_size',
    compute a bounding box that is always the same size (i.e., does not
    expand/contract if the eyes move or blink). The bounding box is
    clamped to the image boundaries.
    """
    
    half_width = window_size[0] // 2
    half_height = window_size[1] // 2
    x_center, y_center = center
    x_min = int(max(0, x_center - half_width))
    y_min = int(max(0, y_center - half_height))
    x_max = int(min(frame_shape[1], x_center + half_width))
    y_max = int(min(frame_shape[0], y_center + half_height))
    return x_min, y_min, x_max, y_max

def process_frame(frame, detector, predictor):
    """
    Process a BGR frame (from webcam or WebRTC) and return the cropped eye region.
    Returns the input frame and the cropped eye region (both in BGR).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray)

    if len(dets) == 0:
        return frame, np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)

    for d in dets:
        try:
            left_eye_points, left_eye_center = facial_landmarks(predictor, gray, d, 36, 42)
            right_eye_points, right_eye_center = facial_landmarks(predictor, gray, d, 42, 48)

            angle, eyes_center = eye_orientation(left_eye_center, right_eye_center)
            rotated_frame, gray_rotated, dets_rotated = orient_eyes(frame, detector, eyes_center, angle)

            for d_rotated in dets_rotated:
                eye_points_rotated, _ = facial_landmarks(predictor, gray_rotated, d_rotated, 36, 48)
                x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(
                    eyes_center, rotated_frame.shape, WINDOW_SIZE
                )
                cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]
                resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
                draw_landmarks(resized_frame, eye_points_rotated)
                return frame, resized_frame
        except Exception as e:
            print(f"[process_frame error]: {e}")
            continue

    return frame, np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)


def main():
    try:
        detector, predictor, cap = create_cam()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector(gray)

            for d in dets:
                # Detect eyes and centers
                left_eye_points, left_eye_center = facial_landmarks(predictor, gray, d, 36, 42)
                right_eye_points, right_eye_center = facial_landmarks(predictor, gray, d, 42, 48)

                # Calculate rotation
                angle, eyes_center = eye_orientation(left_eye_center, right_eye_center)

                # Rotate frame based on eyes
                rotated_frame, gray_rotated, dets_rotated = orient_eyes(frame, detector, eyes_center, angle)

                for d_rotated in dets_rotated:
                    eye_points_rotated, _ = facial_landmarks(predictor, gray_rotated, d_rotated, 36, 48)

                    # Crop to fixed bounding box around eyes
                    x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(eyes_center, rotated_frame.shape, WINDOW_SIZE)
                    cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]

                    # Resize for display
                    resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)

                    # Optional: draw landmarks
                    draw_landmarks(resized_frame, eye_points_rotated)

                    # Show the eye region
                    cv2.imshow('Eyes', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
