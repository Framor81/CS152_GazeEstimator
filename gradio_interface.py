import gradio as gr
import cv2
import numpy as np
import dlib
from PreviousEyeBounding import facial_landmarks, eye_orientation, orient_eyes, calculate_fixed_bounding_box
import time
import os
import torch
from swin_model import Swin
import torchvision.transforms as T
from PIL import Image

# Constants
WINDOW_SIZE = (200, 100)
DIRECTIONS = """
Instructions:
1. Position your face in front of the camera
2. Keep your head steady and look at the camera
3. The red dots will track your eye movements
4. Gaze direction will be displayed on the image and below
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

# Get class names - make sure they match your model's classes
class_names = ['closed', 'down', 'left', 'right', 'straight', 'up']

# Define image transforms for model inference
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Load the Swin model
def load_model():
    debug_print("Loading Swin Transformer model")
    try:
        model = Swin()
        # Modify the output layer to match the number of classes in your trained model
        model.layer = torch.nn.Linear(768, len(class_names))
        model.load_state_dict(torch.load("swin_model.pkl", map_location=torch.device('cpu')))
        model.eval()
        debug_print("Model loaded successfully")
        return model
    except Exception as e:
        debug_print(f"ERROR loading model: {str(e)}")
        return None

class EyeTracker:
    def __init__(self):
        self.detector = None
        self.predictor = None
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.current_gaze = "Unknown"
        self.model = None
        debug_print("EyeTracker initialized")
        
        # Try to load model on initialization
        try:
            self.model = load_model()
        except Exception as e:
            debug_print(f"Failed to load model on init: {str(e)}")

    def setup_camera(self):
        """Initialize the camera and detectors"""
        debug_print("Setting up camera and detectors")
        try:
            # Initialize face detector
            self.detector = dlib.get_frontal_face_detector()
            
            # Initialize landmark predictor
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            debug_print("Detector and predictor initialized")
            
            # Initialize camera - try different approaches
            if os.name == 'nt':  # Windows
                debug_print("Trying DirectShow backend on Windows")
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            # If camera not opened, try standard approach
            if self.cap is None or not self.cap.isOpened():
                debug_print("Trying default camera")
                self.cap = cv2.VideoCapture(0)
            
            # Final check
            if self.cap is None or not self.cap.isOpened():
                debug_print("Failed to open camera")
                return False
            
            # Test read
            ret, frame = self.cap.read()
            if not ret:
                debug_print("Camera opened but can't read frames")
                return False
                
            debug_print(f"Camera opened successfully. Size: {frame.shape}")
            return True
        
        except Exception as e:
            debug_print(f"Error in setup_camera: {str(e)}")
            return False

    def start_camera(self):
        """Start the camera"""
        if self.is_running:
            debug_print("Camera already running")
            return True
            
        debug_print("Starting camera")
        success = self.setup_camera()
        if success:
            self.is_running = True
            debug_print("Camera started successfully")
            return True
        else:
            debug_print("Failed to start camera")
            return False

    def stop_camera(self):
        """Stop the camera"""
        debug_print("Stopping camera")
        if self.cap:
            self.cap.release()
        self.cap = None
        self.is_running = False
        debug_print("Camera stopped")

    def predict_gaze(self, eye_frame):
        """Predict gaze direction using the Swin model"""
        try:
            if self.model is None:
                debug_print("Model not loaded, loading now")
                self.model = load_model()
                if self.model is None:
                    return "Model error"
            
            # Convert the frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB))
            
            # Apply transformations
            img_tensor = transform(pil_image).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
            
            # Get the predicted class
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            
            # Ensure the index is valid
            if 0 <= predicted_idx < len(class_names):
                predicted_class = class_names[predicted_idx]
                debug_print(f"Predicted class: {predicted_class}")
                return predicted_class
            else:
                debug_print(f"Invalid prediction index: {predicted_idx}")
                return "Error"
                
        except Exception as e:
            debug_print(f"Error in predict_gaze: {str(e)}")
            return "Error"
    
    def get_test_frame(self):
        """Return a test frame for debugging"""
        frame = np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # BGR format
        cv2.putText(frame, "Camera Error", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def process_frame(self):
        """Process a single frame"""
        debug_print(f"Processing frame {self.frame_count}")
        
        if not self.is_running or self.cap is None:
            debug_print("Camera not running")
            return self.get_test_frame(), "Unknown"
        
        try:
            # Read a frame
            ret, frame = self.cap.read()
            if not ret:
                debug_print("Failed to read frame")
                return self.get_test_frame(), "Unknown"
            
            self.frame_count += 1
            debug_print(f"Frame {self.frame_count} read. Shape: {frame.shape}")
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            dets = self.detector(gray)
            if len(dets) > 0:
                rect = dets[0]  # Use the first detected face
                debug_print(f"Face detected at: {rect}")
            else:
                # Create artificial face region
                h, w = gray.shape
                rect = dlib.rectangle(
                    left=int(w*0.2), 
                    top=int(h*0.2), 
                    right=int(w*0.8), 
                    bottom=int(h*0.6)
                )
                debug_print("Using artificial face region")
            
            # Detect eye landmarks
            left_eye_points, left_eye_center = facial_landmarks(self.predictor, gray, rect, 36, 42)
            right_eye_points, right_eye_center = facial_landmarks(self.predictor, gray, rect, 42, 48)
            
            # Calculate orientation
            angle, eyes_center = eye_orientation(left_eye_center, right_eye_center)
            
            # Rotate frame
            rotated_frame, gray_rotated, _ = orient_eyes(frame, self.detector, eyes_center, angle)
            
            # Get eye landmarks in rotated frame
            left_eye_points_rotated, _ = facial_landmarks(self.predictor, gray_rotated, rect, 36, 42)
            right_eye_points_rotated, _ = facial_landmarks(self.predictor, gray_rotated, rect, 42, 48)
            eye_points_rotated = np.vstack((left_eye_points_rotated, right_eye_points_rotated))
            
            # Crop to eyes region
            x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(eyes_center, rotated_frame.shape, WINDOW_SIZE)
            
            # Check for valid crop dimensions
            if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or \
               x_max > rotated_frame.shape[1] or y_max > rotated_frame.shape[0]:
                debug_print("Invalid bounding box dimensions")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "Unknown"
            
            # Crop and resize
            cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]
            resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
            
            # Draw landmarks
            for (x, y) in eye_points_rotated:
                if x_min <= x < x_max and y_min <= y < y_max:
                    x_adj = int((x - x_min) * WINDOW_SIZE[0] / (x_max - x_min))
                    y_adj = int((y - y_min) * WINDOW_SIZE[1] / (y_max - y_min))
                    if 0 <= x_adj < WINDOW_SIZE[0] and 0 <= y_adj < WINDOW_SIZE[1]:
                        cv2.circle(resized_frame, (x_adj, y_adj), 2, (0, 0, 255), -1)
            
            # Predict gaze
            predicted_gaze = self.predict_gaze(resized_frame)
            if predicted_gaze not in ["Error", "Model error"]:
                self.current_gaze = predicted_gaze
            
            # Add text to image
            cv2.putText(resized_frame, f"Gaze: {self.current_gaze}", (10, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Convert to RGB for Gradio
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            debug_print(f"Frame processed, gaze: {self.current_gaze}")
            return rgb_frame, self.current_gaze
            
        except Exception as e:
            debug_print(f"Error in process_frame: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())
            return self.get_test_frame(), "Error"

# Create a global instance of the tracker
debug_print("Creating tracker instance")
tracker = EyeTracker()

# For the Gradio interface
def toggle_camera(is_running):
    debug_print(f"Toggle camera called with is_running={is_running}")
    if not is_running:
        success = tracker.start_camera()
        if success:
            return "Stop Camera", True, "Camera running"
        else:
            return "Start Camera (Failed)", False, "Failed to start camera"
    else:
        tracker.stop_camera()
        return "Start Camera", False, "Camera stopped"

def update_frame():
    debug_print("Update frame called")
    frame, gaze = tracker.process_frame()
    return frame, gaze

def webcam_feed():
    debug_print("Webcam feed started")
    try:
        while True:
            if tracker.is_running:
                frame, _ = tracker.process_frame()
                yield frame
            else:
                yield tracker.get_test_frame()
            time.sleep(0.1)  # 10 FPS
    except Exception as e:
        debug_print(f"Error in webcam_feed: {str(e)}")
        yield tracker.get_test_frame()

# Create the Gradio interface
with gr.Blocks(title="Eye Gaze Detection", theme=gr.themes.Base()) as demo:
    debug_print("Creating Gradio interface")
    gr.Markdown("## Eye Gaze Direction Detection")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(DIRECTIONS)
            camera_state = gr.State(value=False)
            toggle_btn = gr.Button("Start Camera", variant="primary")
            status_text = gr.Textbox(label="Status", value="Camera not started")
            gaze_text = gr.Textbox(label="Gaze Direction", value="Unknown")
            
        with gr.Column(scale=2):
            # Version 1: Manual refresh
            eye_image = gr.Image(label="Eye Region")
            refresh_btn = gr.Button("Manual Refresh")
            
            # Version 2: Streaming
            stream_image = gr.Image(label="Streaming Eye Region", streaming=True)
    
    # Connect the buttons to functions
    toggle_btn.click(
        fn=toggle_camera,
        inputs=[camera_state],
        outputs=[toggle_btn, camera_state, status_text]
    )
    
    refresh_btn.click(
        fn=update_frame,
        inputs=[],
        outputs=[eye_image, gaze_text]
    )
    
    # Set up streaming
    stream_image.stream(webcam_feed)
    
    # Set up timer for gaze text updates
    demo.load(None, None, None, js="""
        function() {
            setInterval(function() {
                if (document.querySelector('button[value="Manual Refresh"]')) {
                    document.querySelector('button[value="Manual Refresh"]').click();
                }
            }, 200);
        }
    """)
    
    debug_print("Gradio interface setup complete")

# Launch the app
if __name__ == "__main__":
    debug_print("Launching application...")
    try:
        
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        
        # Launch with queue
        demo.queue()
        demo.launch(debug=True)
        debug_print("App launched successfully")
    except Exception as e:
        debug_print(f"Error launching app: {str(e)}")
    debug_print("End of script reached")