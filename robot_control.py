import time

import qwiic_scmd

# Constants
LEFT_MOTOR = 1
RIGHT_MOTOR = 0
FORWARD_SPEED = 200
TURN_SPEED = 180
REVERSE_SPEED = -200

def initialize_motors():
    motors = qwiic_scmd.QwiicScmd()
    if not motors.connected:
        print("Motor driver not connected!")
        return None
    print("Motor driver connected.")
    motors.enable()
    return motors

def stop_motors(motors):
    motors.set_drive(LEFT_MOTOR, 1, 0)
    motors.set_drive(RIGHT_MOTOR, 1, 0)

def handle_gaze_direction(motors, direction):
    direction = direction.lower()
    print(f"Gaze direction: {direction}")

    if direction == "left":
        # Turn left (right wheel forward, left wheel slow or reverse)
        motors.set_drive(LEFT_MOTOR, 1, 0)
        motors.set_drive(RIGHT_MOTOR, 1, TURN_SPEED)

    elif direction == "right":
        # Turn right (left wheel forward, right wheel slow or reverse)
        motors.set_drive(LEFT_MOTOR, 1, TURN_SPEED)
        motors.set_drive(RIGHT_MOTOR, 1, 0)

    elif direction == "up":
        # Move forward
        motors.set_drive(LEFT_MOTOR, 1, FORWARD_SPEED)
        motors.set_drive(RIGHT_MOTOR, 1, FORWARD_SPEED)

    elif direction == "down":
        # Move backward (set direction to 0 for reverse)
        motors.set_drive(LEFT_MOTOR, 0, -REVERSE_SPEED)
        motors.set_drive(RIGHT_MOTOR, 0, -REVERSE_SPEED)

    elif direction == "straight":
        # Idle (stop motors)
        stop_motors(motors)

    elif direction == "closed":
        # Run motors (go forward aggressively)
        motors.set_drive(LEFT_MOTOR, 1, 255)
        motors.set_drive(RIGHT_MOTOR, 1, 255)

    else:
        # Unknown command, stop for safety
        stop_motors(motors)

# Example loop to simulate gaze inputs
if __name__ == "__main__":
    motors = initialize_motors()
    if motors is None:
        exit(1)

    try:
        directions = ["left", "right", "up", "down", "straight", "closed"]

        for dir in directions:
            handle_gaze_direction(motors, dir)
            time.sleep(2)
            stop_motors(motors)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted. Stopping motors.")
        stop_motors(motors)
        motors.disable()
