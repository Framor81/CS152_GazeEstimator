import time

import cv2
import gradio as gr
import numpy as np

from PreviousEyeBounding import (WINDOW_SIZE, calculate_fixed_bounding_box,
                                 create_cam, draw_landmarks, eye_orientation,
                                 facial_landmarks, orient_eyes)

print("Starting application...")
print("Initializing camera and models...")

# Initialize the camera and models
try:
    detector, predictor, cap = create_cam()
    print("Camera and models initialized successfully")
    print(f"Camera is opened: {cap.isOpened()}")
    
    # Test camera read
    ret, test_frame = cap.read()
    if ret:
        print("Successfully read test frame from camera")
    else:
        print("Failed to read test frame from camera")
except Exception as e:
    print(f"Error initializing camera and models: {str(e)}")
    raise

def process_frame(frame):
    """
    Process a single frame to detect and display eye regions
    Returns both the original frame and the eye region
    """
    try:
        print("Processing new frame...")
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)
        
        if len(dets) == 0:
            print("No face detected in frame")
            return frame, np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        
        print(f"Detected {len(dets)} faces")
        
        # Process each detected face
        for d in dets:
            try:
                # Detect eyes and centers
                left_eye_points, left_eye_center = facial_landmarks(predictor, gray, d, 36, 42)
                right_eye_points, right_eye_center = facial_landmarks(predictor, gray, d, 42, 48)
                
                print("Eye centers detected")
                
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
                    
                    # Draw landmarks
                    draw_landmarks(resized_frame, eye_points_rotated)
                    
                    print("Successfully processed eye region")
                    return frame, resized_frame
                    
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue
                
        return frame, np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return frame, np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)

def get_webcam_feed():
    """
    Capture frame from webcam and process it
    """
    print("Attempting to read frame from webcam...")
    if not cap.isOpened():
        print("Error: Camera is not opened!")
        return None, None
        
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam")
        return None, None
    
    print("Successfully captured frame from webcam")
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Process the frame
    original_frame, eye_region = process_frame(frame)
    
    # Convert BGR to RGB for Gradio display
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
    
    return original_frame, eye_region

# Create Gradio interface
print("Creating Gradio interface...")
with gr.Blocks() as demo:
    gr.Markdown("# Eye Tracking Demo")
    gr.Markdown("This demo shows the webcam feed and the detected eye region in real-time.")
    
    with gr.Row():
        with gr.Column():
            original_output = gr.Image(label="Original Webcam Feed")
        with gr.Column():
            eye_output = gr.Image(label="Eye Region")
    
    # Initialize timer with proper parameters
    print("Setting up timer...")
    timer = gr.Timer(0.01)  # 100ms interval
    
    # Set up the timer to update the webcam feed
    print("Setting up timer tick event...")
    timer.tick(
        fn=get_webcam_feed,
        outputs=[original_output, eye_output]
    )

if __name__ == "__main__":
    try:
        print("Starting Gradio interface...")
        demo.launch()
    except Exception as e:
        print(f"Error launching Gradio interface: {str(e)}")
    finally:
        print("Releasing camera resources...")
        cap.release()
        print("Camera released") 