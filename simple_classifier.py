import os

import cv2
import dlib
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms as T
from PIL import Image

from swin_model import Swin

# Constants
WINDOW_SIZE = (120, 100)
class_names = ['down', 'left', 'right', 'straight', 'up', "closed"]

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define image transforms for model inference
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def facial_landmarks(predictor, face, landmarks, lB, uB):
    """Get facial landmarks for eye region"""
    shape = predictor(face, landmarks)
    eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(lB, uB)], np.int32)
    eye_center = np.mean(eye_points, axis=0).astype(int)
    return eye_points, eye_center

def calculate_fixed_bounding_box(center, frame_shape, window_size):
    """Calculate fixed size bounding box centered on eye"""
    half_width = window_size[0] // 2
    half_height = window_size[1] // 2
    x_center, y_center = center
    x_min = int(max(0, x_center - half_width))
    y_min = int(max(0, y_center - half_height))
    x_max = int(min(frame_shape[1], x_center + half_width))
    y_max = int(min(frame_shape[0], y_center + half_height))
    return x_min, y_min, x_max, y_max

def load_model():
    """Load the Swin Transformer model"""
    try:
        print("Loading Swin model...")
        
        # Check if model file exists
        model_path = "swin_model.pkl"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        print(f"Found model file at {model_path}")
        
        # Initialize model
        model = Swin()
        print("Swin model initialized")
        
        # Initialize the classification layer (using 'layer' instead of 'head' to match saved state)
        model.layer = torch.nn.Linear(768, len(class_names))
        print("Classification layer initialized")
        
        # Load the state dict
        print("Loading state dict...")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"State dict keys: {state_dict.keys()}")
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        print("State dict loaded successfully")
        
        # Set to eval mode
        model.eval()
        print("Model set to eval mode")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None

def classify_image(image):
    """Classify an image using the Swin Transformer model"""
    try:
        if image is None:
            return "No image uploaded", None, None, None
            
        # Load model if not already loaded
        if not hasattr(classify_image, 'model'):
            print("Loading model for first time...")
            classify_image.model = load_model()
            if classify_image.model is None:
                return "Failed to load model", None, None, None
        
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to BGR if it's RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Try to detect face and get eye region
        dets = detector(gray)
        if len(dets) > 0:
            # Process first detected face
            rect = dets[0]
            
            # Get left eye landmarks
            left_eye_points, left_eye_center = facial_landmarks(predictor, gray, rect, 36, 42)
            
            # Calculate bounding box
            x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(left_eye_center, image.shape, WINDOW_SIZE)
            
            # Check for valid crop dimensions
            if x_min < x_max and y_min < y_max and x_min >= 0 and y_min >= 0 and \
               x_max <= image.shape[1] and y_max <= image.shape[0]:
                # Crop and resize
                cropped_frame = image[y_min:y_max, x_min:x_max]
                resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
                
                # Draw rectangle on original image
                result_image = image.copy()
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(result_image, "Eye Region Detected", (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convert to PIL Image for model input
                pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                cropped_eye = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            else:
                # If eye region is invalid, use whole image
                print("Invalid eye region, using whole image")
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                result_image = image.copy()
                cv2.putText(result_image, "Using Whole Image", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cropped_eye = None
        else:
            # No face detected, use whole image
            print("No face detected, using whole image")
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result_image = image.copy()
            cv2.putText(result_image, "No Face Detected - Using Whole Image", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cropped_eye = None
        
        # Apply transformations
        img_tensor = transform(pil_image).unsqueeze(0)
        print(f"Input tensor shape: {img_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            print("Making prediction...")
            outputs = classify_image.model(img_tensor)
            print(f"Model output shape: {outputs.shape}")
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_idx = predicted.item()
            confidence_val = confidence.item()
            
            # Get all class probabilities
            all_probs = probs[0].cpu().numpy()
            print(f"Class probabilities: {dict(zip(class_names, all_probs))}")
            
            if 0 <= predicted_idx < len(class_names):
                predicted_class = class_names[predicted_idx]
            else:
                predicted_class = "Unknown"
        
        # Create bar plot using plotly
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=all_probs,
                marker_color=['red' if i == predicted_idx else 'blue' for i in range(len(class_names))]
            )
        ])
        
        # Update layout
        fig.update_layout(
            title="Confidence Scores",
            xaxis_title="Gaze Direction",
            yaxis_title="Confidence",
            yaxis_range=[0, 1],
            showlegend=False
        )
        
        # Add prediction text to result image
        cv2.putText(result_image, f"Gaze: {predicted_class}", (10, result_image.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to RGB for display
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        return f"Predicted: {predicted_class} (Confidence: {confidence_val:.2f})", fig, result_image, cropped_eye
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}", None, None, None

# Create the Gradio interface
with gr.Blocks(title="Eye Gaze Classifier") as demo:
    gr.Markdown("# Eye Gaze Classification")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(label="Input Image", type="pil", sources=["upload", "webcam"])
            classify_btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=1):
            result_text = gr.Textbox(label="Prediction")
            result_plot = gr.Plot(label="Confidence Scores")
            result_image = gr.Image(label="Detected Eye Region", height=200)
            cropped_eye = gr.Image(label="Cropped Eye", height=200)
    
    classify_btn.click(
        fn=classify_image,
        inputs=[input_image],
        outputs=[result_text, result_plot, result_image, cropped_eye]
    )

if __name__ == "__main__":
    demo.launch() 