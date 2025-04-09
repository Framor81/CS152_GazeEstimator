import gradio as gr
import cv2
import numpy as np
import dlib
from PreviousEyeBounding import facial_landmarks, eye_orientation, orient_eyes, calculate_fixed_bounding_box
import time
import sys
import os

# Constants
WINDOW_SIZE = (200, 100)
DIRECTIONS = """
Instructions:
1. Position your face in front of the camera
2. Keep your head steady and look at the camera
3. The red dots will track your eye movements
"""

print("Starting Robo-vision application...", flush=True)

# Add a debug flag to control verbose output
DEBUG = True

def debug_print(message):
    """Helper function to print debug messages only when DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}", flush=True)
    else:
        # Still print critical messages
        if message.startswith("ERROR"):
            print(f"CRITICAL: {message}", flush=True)

debug_print("Debug printing enabled")

# Simple camera creation function without contrast adjustments
def create_cam_simple():
    """Create a camera with basic dlib detector setup"""
    debug_print("Setting up camera with original approach")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
    
    # Try to open camera with DirectShow backend on Windows
    if os.name == 'nt':  # Windows
        debug_print("Using DirectShow backend on Windows")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        debug_print("Using default backend")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        debug_print("Failed to open camera")
        return detector, predictor, None
    
    debug_print("Camera opened successfully")
    return detector, predictor, cap

class EyeTracker:
    def __init__(self):
        self.detector = None
        self.predictor = None
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        debug_print("EyeTracker initialized")

    def toggle_camera(self, is_running):
        debug_print(f"Toggle camera called with is_running={is_running}")
        if not is_running:
            debug_print("Starting camera...")
            try:
                self.detector, self.predictor, self.cap = create_cam_simple()
                if self.cap and self.cap.isOpened():
                    debug_print(f"Camera opened successfully. Width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                else:
                    debug_print("Camera opened but may not be working properly")
                
                self.is_running = True
                debug_print("Camera started successfully")
                return "Stop Camera", True
            except Exception as e:
                debug_print(f"ERROR starting camera: {str(e)}")
                return "Start Camera (Failed)", False
        else:
            debug_print("Stopping camera...")
            if self.cap is not None:
                self.cap.release()
            self.detector = None
            self.predictor = None
            self.cap = None
            self.is_running = False
            debug_print("Camera stopped")
            return "Start Camera", False

    def get_test_frame(self):
        """Return a test frame (colored rectangle) for debugging"""
        debug_print("Generating test frame")
        # Create a simple colored frame for testing
        frame = np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        # Fill with a blue color
        frame[:, :] = (255, 0, 0)  # BGR format
        # Add frame number text
        self.frame_count += 1
        cv2.putText(frame, f"Frame {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Add message about camera issues
        cv2.putText(frame, "Camera Error", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame

    def process_frame(self):
        """Process a frame to extract eye region without requiring face detection"""
        if not self.is_running or self.cap is None:
            debug_print("Camera not running in process_frame")
            # Return a test frame instead of None for debugging
            return self.get_test_frame()

        try:
            ret, frame = self.cap.read()
            if not ret:
                debug_print("ERROR: Failed to read frame from camera")
                return self.get_test_frame()

            # Print frame dimensions for debugging
            debug_print(f"Frame {self.frame_count} read successfully: {frame.shape}")
            self.frame_count += 1
            
            # Following the same basic processing without requiring face detection first
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create artificial face region covering center of frame
            h, w = gray.shape
            rect = dlib.rectangle(left=int(w*0.2), top=int(h*0.2), 
                                 right=int(w*0.8), bottom=int(h*0.8))
            
            try:
                # Detect eyes and centers using the artificial face region
                left_eye_points, left_eye_center = facial_landmarks(self.predictor, gray, rect, 36, 42)
                right_eye_points, right_eye_center = facial_landmarks(self.predictor, gray, rect, 42, 48)
                debug_print(f"Eye centers detected - Left: {left_eye_center}, Right: {right_eye_center}")
                
                # Calculate rotation
                angle, eyes_center = eye_orientation(left_eye_center, right_eye_center)
                debug_print(f"Eye orientation - Angle: {angle:.2f}, Center: {eyes_center}")

                # Rotate frame based on eyes
                rotated_frame, gray_rotated, _ = orient_eyes(frame, self.detector, eyes_center, angle)
                
                # Use original rect for landmark detection on rotated frame
                rotated_rect = rect  # Use same region in rotated frame
                
                # Crop to fixed bounding box around eyes
                x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(eyes_center, rotated_frame.shape, WINDOW_SIZE)
                debug_print(f"Bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                
                # Check for valid crop dimensions
                if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or \
                   x_max > rotated_frame.shape[1] or y_max > rotated_frame.shape[0]:
                    debug_print(f"Invalid bounding box dimensions, returning original frame")
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]
                debug_print(f"Cropped frame shape: {cropped_frame.shape}")

                # Resize for display
                resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
                debug_print(f"Frame resized to {WINDOW_SIZE}")

                # Get eye landmarks in rotated frame
                eye_points_rotated, _ = facial_landmarks(self.predictor, gray_rotated, rotated_rect, 36, 48)
                
                # Draw landmarks
                for (x, y) in eye_points_rotated:
                    # Check if the point is inside the cropped area
                    if x_min <= x < x_max and y_min <= y < y_max:
                        # Adjust coordinates to the resized frame
                        x_adj = int((x - x_min) * WINDOW_SIZE[0] / (x_max - x_min))
                        y_adj = int((y - y_min) * WINDOW_SIZE[1] / (y_max - y_min))
                        
                        # Make sure coordinates are within bounds
                        if 0 <= x_adj < WINDOW_SIZE[0] and 0 <= y_adj < WINDOW_SIZE[1]:
                            cv2.circle(resized_frame, (x_adj, y_adj), 2, (0, 0, 255), -1)

                # Convert to RGB for Gradio
                debug_print("Returning processed eye region frame")
                return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
            except Exception as e:
                debug_print(f"Error processing eye landmarks: {str(e)}")
                import traceback
                debug_print(traceback.format_exc())
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            debug_print(f"ERROR in process_frame: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())
            return self.get_test_frame()


# Create a global instance of the tracker
debug_print("Creating tracker instance")
tracker = EyeTracker()

# This is the streaming function that works with gr.Image().stream()
def webcam_feed():
    debug_print("Webcam feed generator started")
    while True:
        try:
            frame = tracker.process_frame()
            if frame is not None:
                yield frame
            else:
                debug_print("Received None frame, skipping")
                time.sleep(0.1)
                continue
            
            # Sleep to control frame rate
            time.sleep(0.1)  # 10 FPS
            
        except Exception as e:
            debug_print(f"Error in webcam_feed: {e}")
            # Create an error frame
            error_frame = np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
            error_frame[:, :] = (0, 0, 255)  # Blue test frame
            cv2.putText(error_frame, f"Error: {str(e)[:20]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            yield error_frame
            time.sleep(0.5)  # Longer sleep after error

with gr.Blocks(title="Robo-vision", theme=gr.themes.Base()) as demo:
    debug_print("Creating Gradio blocks")
    gr.Markdown("## Robo-vision")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(DIRECTIONS)
            camera_running = gr.State(value=False)
            toggle_btn = gr.Button("Start Camera", variant="primary")
            debug_print("Button created")
            
            # Add a manual refresh button for testing
            refresh_btn = gr.Button("Refresh View", variant="secondary")
            
            # Add status text
            status_text = gr.Textbox(label="Status", value="Camera not started")
            
        with gr.Column(scale=2):
            # Use a regular Image component for static updates
            image_output = gr.Image(label="Camera View (Static)")
            debug_print("Static image component created")
            
            # Streaming component with correct method
            camera_output = gr.Image(label="Eye Region (Streaming)", streaming=True)
            debug_print("Camera output component created")
    
    # Add update event for the static image
    def update_static_image():
        debug_print("Updating static image")
        return tracker.process_frame()
    
    refresh_btn.click(
        fn=update_static_image,
        inputs=[],
        outputs=[image_output]
    )
    
    # Modified toggle function with status
    def toggle_with_status(is_running):
        btn_text, new_running = tracker.toggle_camera(is_running)
        if new_running:
            status = "Camera running - Position your face in front of the camera"
        else:
            status = "Camera stopped"
        return btn_text, new_running, status
    
    toggle_btn.click(
        fn=toggle_with_status,
        inputs=[camera_running],
        outputs=[toggle_btn, camera_running, status_text]
    )
    
    # Set up event to update static image when camera toggles
    def update_after_toggle(btn_text, is_running):
        debug_print(f"Camera state changed to {is_running}, updating static image")
        time.sleep(0.5)  # Small delay to let camera initialize
        if is_running:
            return tracker.process_frame()
        else:
            # Create a black image when camera is off
            black = np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
            return black
    
    toggle_btn.click(
        fn=update_after_toggle,
        inputs=[toggle_btn, camera_running],
        outputs=[image_output]
    )

    # Connect the webcam feed to the streaming component
    camera_output.stream(webcam_feed)
    debug_print("Camera stream set up successfully")
        
    debug_print("Gradio interface setup complete")


if __name__ == "__main__":
    debug_print("Launching Robo-vision demo...")
    try:
        # Force OpenCV to use DirectShow on Windows
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Lower priority for Media Foundation
        os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "100"  # Higher priority for DirectShow
        
        # Optimize OpenCV performance
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        
        # Launch with queue workers to handle concurrent requests
        demo.queue()
        demo.launch(debug=True)
        debug_print("Demo launched successfully")
    except Exception as e:
        debug_print(f"ERROR launching demo: {str(e)}")
    debug_print("End of script reached")