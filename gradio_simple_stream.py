import cv2
import gradio as gr
import numpy as np
from gradio_webrtc import WebRTC


def process_webcam_frame(frame):
    """
    Process each frame from the webcam
    frame: RGB image from browser webcam
    returns: RGB image for display
    """
    if frame is None:
        return None
    
    try:
        # Convert from RGB to BGR for OpenCV processing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add your processing here
        # For example:
        # - Face detection
        # - Eye tracking
        # - Image transformations
        
        # Convert back to RGB for display
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

# WebRTC configuration
rtc_configuration = {
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"],
        }
    ]
}

# Define the display size
DISPLAY_SIZE = (320, 240)  # Smaller size for the display

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# Webcam Stream Demo")
    
    # WebRTC component with fixed size
    webcam = WebRTC(
        label="Webcam Feed",
        rtc_configuration=rtc_configuration,
        width=DISPLAY_SIZE[0],
        height=DISPLAY_SIZE[1]
    )
    
    # Set up the streaming
    webcam.stream(
        fn=process_webcam_frame,
        inputs=[webcam],
        outputs=[webcam]
    )

if __name__ == "__main__":
    demo.launch() 