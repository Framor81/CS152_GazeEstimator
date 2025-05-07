import os
import time

import cv2
import dlib
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from PreviousEyeBounding import (calculate_fixed_bounding_box, eye_orientation,
                                 facial_landmarks, orient_eyes)
from swin_model import Swin


class EyeCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(EyeCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

# Constants
WINDOW_SIZE = (100, 50)
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

# Get class names 
class_names = [ 'down', 'left', 'right', 'straight', 'up']

# Define image transforms for model inference
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Define image transforms for CNN model
cnn_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#  load_model function
def load_model(model):
    debug_print(f"Loading {'CNN' if model == 'CNN' else 'Swin Transformer'} model")
    try:
        if model == "CNN":
            # Load CNN model
            model = EyeCNN(num_classes=len(class_names))
            model.load_state_dict(torch.load("CNN_model.pkl", map_location=torch.device('cpu')))
        else:
            class_names.append('closed')
            # Load Swin Transformer model
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
        self.current_confidence = 0.0
        self.model = None
        debug_print("EyeTracker initialized")
        
        # Try to load model on initialization
        try:
            self.model = load_model(model='swin') # CCN vs Swin
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
        """Predict gaze direction using the loaded model"""
        try:
            if self.model is None:
                debug_print("Model not loaded, loading now")
                self.model = load_model(use_cnn=True)  # Using CNN model
                if self.model is None:
                    return "Model error", 0.0
            
            # Convert the frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB))
            
            # For CNN model, we need to use the correct image size (224x224)
            # Resize image directly to 224x224 for CNN model
            pil_image = pil_image.resize((224, 224))
            
            # Apply transformations
            if isinstance(self.model, EyeCNN):
                img_tensor = cnn_transform(pil_image).unsqueeze(0)
            else:
                img_tensor = transform(pil_image).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
            
            # Get the predicted class
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_idx = predicted.item()
            confidence_val = confidence.item()
            
            # Ensure the index is valid
            if 0 <= predicted_idx < len(class_names):
                predicted_class = class_names[predicted_idx]
                debug_print(f"Predicted class: {predicted_class}, Confidence: {confidence_val:.2f}")
                return predicted_class, confidence_val
            else:
                debug_print(f"Invalid prediction index: {predicted_idx}")
                return "Error", 0.0
                
        except Exception as e:
            debug_print(f"Error in predict_gaze: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())
            return "Error", 0.0
    
    def get_test_frame(self):
        """Return a test frame for debugging"""
        frame = np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # BGR format
        cv2.putText(frame, "Camera Off", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def process_frame(self):
        """Process a single frame"""
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
            
            # Detect left eye landmarks only
            left_eye_points, left_eye_center = facial_landmarks(self.predictor, gray, rect, 36, 42)
            
            # Crop to left eye region
            x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(left_eye_center, frame.shape, WINDOW_SIZE)
            
            # Check for valid crop dimensions
            if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or \
               x_max > frame.shape[1] or y_max > frame.shape[0]:
                debug_print("Invalid bounding box dimensions")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "Unknown"
            
            # Crop and resize
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
            
            # Draw landmarks for left eye
            # draw_landmarks(resized_frame, left_eye_points)
            
            # Predict gaze
            predicted_gaze, confidence = self.predict_gaze(resized_frame)
            if predicted_gaze not in ["Error", "Model error"]:
                self.current_gaze = predicted_gaze
                self.current_confidence = confidence
            
            # Add text to image
            cv2.putText(resized_frame, f"Gaze: {self.current_gaze}", (10, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(resized_frame, f"Conf: {self.current_confidence:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Convert to RGB for Gradio
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            debug_print(f"Frame processed, gaze: {self.current_gaze}, confidence: {self.current_confidence:.2f}")
            return rgb_frame, self.current_gaze, self.current_confidence
            
        except Exception as e:
            debug_print(f"Error in process_frame: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())
            return self.get_test_frame(), "Error"

# Create a global instance of the tracker
debug_print("Creating tracker instance")
tracker = EyeTracker()

# Global variable to control webcam feed
webcam_active = False

# For the Gradio interface
def toggle_camera(is_running):
    global webcam_active
    debug_print(f"Toggle camera called with is_running={is_running}")
    
    if not is_running:
        success = tracker.start_camera()
        if success:
            webcam_active = True  # Set streaming active
            debug_print("Camera and streaming started")
            return "Stop Camera", True, "Camera running"
        else:
            webcam_active = False  # Ensure streaming is off
            debug_print("Failed to start camera")
            return "Start Camera (Failed)", False, "Failed to start camera"
    else:
        tracker.stop_camera()
        webcam_active = False  # Stop streaming
        debug_print("Camera and streaming stopped")
        return "Start Camera", False, "Camera stopped"

def update_frame():
    debug_print("Update frame called")
    if tracker.is_running:
        frame, gaze, confidence = tracker.process_frame()
        return frame, f"{gaze} (Confidence: {confidence:.2f})"
    else:
        return tracker.get_test_frame(), "Camera Off"

def process_webcam_frame():
    debug_print("Process webcam frame called")
    try:
        if webcam_active and tracker.is_running:
            frame, gaze, confidence = tracker.process_frame()
            debug_print(f"Frame processed, gaze: {gaze}, confidence: {confidence:.2f}")
            return frame, f"{gaze} (Confidence: {confidence:.2f})"
        else:
            test_frame = tracker.get_test_frame()
            debug_print("Returning test frame (camera off)")
            return test_frame, "Camera Off"
    except Exception as e:
        debug_print(f"Error in process_webcam_frame: {str(e)}")
        return tracker.get_test_frame(), "Error"

def process_uploaded_image(image):
    """Process an uploaded image using the Swin model"""
    try:
        if image is None:
            return None, "No image uploaded"
            
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to BGR if it's RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        dets = tracker.detector(gray)
        if len(dets) == 0:
            return image, "No face detected in image"
            
        # Process first detected face
        rect = dets[0]
        
        # Get left eye landmarks
        left_eye_points, left_eye_center = facial_landmarks(tracker.predictor, gray, rect, 36, 42)
        
        # Calculate bounding box
        x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(left_eye_center, image.shape, WINDOW_SIZE)
        
        # Check for valid crop dimensions
        if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or \
           x_max > image.shape[1] or y_max > image.shape[0]:
            return image, "Invalid eye region in image"
        
        # Crop and resize
        cropped_frame = image[y_min:y_max, x_min:x_max]
        resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
        
        # Predict gaze
        predicted_gaze, confidence = tracker.predict_gaze(resized_frame)
        
        # Draw results on original image
        result_image = image.copy()
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(result_image, f"Gaze: {predicted_gaze}", (x_min, y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(result_image, f"Conf: {confidence:.2f}", (x_min, y_min-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to RGB for display
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return result_image, f"Predicted Gaze: {predicted_gaze} (Confidence: {confidence:.2f})"
        
    except Exception as e:
        debug_print(f"Error processing uploaded image: {str(e)}")
        return image, f"Error processing image: {str(e)}"

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
            # Single image component for displaying frames
            eye_image = gr.Image(label="Eye Region (Live Feed)")
            debug_print("Created image component")
            
            # Add a refresh button that will be auto-clicked by JavaScript
            refresh_btn = gr.Button("Refresh", visible=False, elem_id="refresh-trigger")
    
    # Add a new section for image upload
    gr.Markdown("## Upload Image for Gaze Detection")
    with gr.Row():
        with gr.Column():
            upload_image = gr.Image(label="Upload Image", type="pil")
            upload_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            result_image = gr.Image(label="Analysis Result")
            result_text = gr.Textbox(label="Prediction Result")
    
    # Connect the toggle button to the toggle_camera function
    toggle_btn.click(
        fn=toggle_camera,
        inputs=[camera_state],
        outputs=[toggle_btn, camera_state, status_text]
    )
    
    # Connect the refresh button to update the frame
    refresh_btn.click(
        fn=process_webcam_frame,
        inputs=[],
        outputs=[eye_image, gaze_text]
    )
    
    # Connect the upload button to process the uploaded image
    upload_btn.click(
        fn=process_uploaded_image,
        inputs=[upload_image],
        outputs=[result_image, result_text]
    )
    
    # Set up auto-refresh using JavaScript, but controlled by camera state
    demo.load(None, None, None, js="""
        function() {
            console.log("Setting up event listener for camera toggle");
            
            // Global variable to store the interval ID
            window.refreshInterval = null;
            
            // Function to start auto-refresh
            window.startAutoRefresh = function() {
                console.log("Starting auto-refresh");
                if (!window.refreshInterval) {
                    window.refreshInterval = setInterval(function() {
                        if (document.getElementById('refresh-trigger')) {
                            document.getElementById('refresh-trigger').click();
                            console.log("Auto-clicked refresh button");
                        }
                    }, 1000);  // Adjust the interval as needed (500ms)
                }
            };
            
            // Function to stop auto-refresh
            window.stopAutoRefresh = function() {
                console.log("Stopping auto-refresh");
                if (window.refreshInterval) {
                    clearInterval(window.refreshInterval);
                    window.refreshInterval = null;
                }
            };
            
            // Monitor button changes to detect camera toggle
            setInterval(function() {
                const button = document.querySelector('button.primary');
                if (button && button.textContent.includes("Stop Camera")) {
                    // Camera is on, start auto-refresh if not already running
                    if (!window.refreshInterval) {
                        window.startAutoRefresh();
                    }
                } else {
                    // Camera is off, stop auto-refresh
                    window.stopAutoRefresh();
                }
            }, 500);  // Check every 500ms
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
        demo.launch(debug=True, share=True)
        debug_print("App launched successfully")
    except Exception as e:
        debug_print(f"Error launching app: {str(e)}")
    debug_print("End of script reached")