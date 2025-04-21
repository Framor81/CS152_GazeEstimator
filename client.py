import cv2
import requests
import base64
import numpy as np
import time

# Simple direct API endpoint
SERVER_URL = "http://localhost:9994/process_frame"

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # Try DirectShow backend on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
if not cap.isOpened():
    print("Failed to open webcam. Exiting.")
    exit(1)

print("Webcam opened successfully!")
print("Controls:")
print("- Arrow keys: manually set gaze direction")
print("- Q: Quit the application")

# Track the last pressed key
last_key = None
current_gaze = "Unknown"

# Main loop
try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            time.sleep(0.2)
            continue
        
        # Show original frame
        cv2.imshow("Original", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Check for arrow keys
        key_command = None
        if key == ord('q'):  # Quit
            break
        elif key == 82 or key == ord('w'):  # Up arrow or W
            key_command = "up"
            last_key = key
        elif key == 84 or key == ord('s'):  # Down arrow or S
            key_command = "down"
            last_key = key
        elif key == 81 or key == ord('a'):  # Left arrow or A
            key_command = "left"
            last_key = key
        elif key == 83 or key == ord('d'):  # Right arrow or D
            key_command = "right"
            last_key = key
        elif key == 32:  # Spacebar
            key_command = "straight"
            last_key = key
        elif key == ord('c'):  # C key
            key_command = "closed"
            last_key = key
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare payload with both image and keyboard command
        payload = {
            "image": f"data:image/jpeg;base64,{img_str}"
        }
        
        # Add keyboard command if one was pressed
        if key_command:
            payload["key_command"] = key_command
            print(f"Sending keyboard command: {key_command}")
        
        # Send to server
        try:
            response = requests.post(
                SERVER_URL,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Get gaze direction
                if 'gaze' in result:
                    current_gaze = result['gaze']
                
                # Display processed image
                if 'image' in result:
                    img_data = base64.b64decode(result['image'].split(',')[1])
                    img_array = np.frombuffer(img_data, np.uint8)
                    processed_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    # Add keyboard control indicator to the image
                    if key_command:
                        print(f"Manual: {key_command}")
                        cv2.putText(processed_img, f"Manual: {key_command}", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    if processed_img is not None:
                        cv2.imshow("Processed Eyes", processed_img)
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()