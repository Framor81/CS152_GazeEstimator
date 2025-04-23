import cv2
import dlib
import gradio as gr
import numpy as np
from gradio_webrtc import WebRTC

from PreviousEyeBounding import create_cam, process_frame

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

# Frame buffer counter
frame_counter = 0
BUFFER_SIZE = 4  # Process every 4th frame

def process_webcam_frame(frame, record):
    """
    Process each frame from the webcam
    frame: RGB image from browser webcam
    record: boolean indicating if recording is enabled
    returns: RGB image for display
    """
    global frame_counter
    
    if frame is None:
        return None
    
    try:
        # Convert from RGB to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if record:
            # Only process every 4th frame
            frame_counter = (frame_counter + 1) % BUFFER_SIZE
            if frame_counter == 0:
                _, eye_region = process_frame(frame_bgr, detector, predictor)
                return cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
            else:
                # Return the last processed frame
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error in webcam processor: {str(e)}")
        return None

# WebRTC configuration
rtc_configuration = {
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"],
        }
    ]
}

# Define the display sizes
NORMAL_SIZE = (520, 480)  # Normal size
RECORDING_SIZE = (600, 300)  # Size when recording

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# Webcam Stream Demo")
    
    # Recording toggle
    record = gr.Checkbox(label="Record/Process Frames")
    
    # WebRTC component with dynamic size
    webcam = WebRTC(
        label="Webcam Feed",
        rtc_configuration=rtc_configuration,
        width=NORMAL_SIZE[0],
        height=NORMAL_SIZE[1]
    )
    
    # Function to update size based on recording state
    def update_size(record_state):
        if record_state:
            return {"width": RECORDING_SIZE[0], "height": RECORDING_SIZE[1]}
        else:
            return {"width": NORMAL_SIZE[0], "height": NORMAL_SIZE[1]}
    
    # Set up the streaming and size updates
    webcam.stream(
        fn=process_webcam_frame,
        inputs=[webcam, record],
        outputs=[webcam]
    )
    
    # Update size when recording state changes
    record.change(
        fn=update_size,
        inputs=[record],
        outputs=[webcam]
    )

if __name__ == "__main__":
    demo.launch() 