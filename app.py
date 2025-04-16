import warnings
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque
import time
from flask import Flask, render_template, Response
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but")

# --- Flask App Initialization ---
app = Flask(__name__)



# --- Global Variables ---
MODEL_PATH = 'svm_sign_model_normalized.pkl'
clf = None
scaler = None # Set to None if your .pkl file only contains the classifier
mp_hands = None
hands = None
prediction_buffer = None
stable_prediction_global = "Initializing..." # Initial state

# --- Prediction Smoothing Parameters ---
PREDICTION_BUFFER_SIZE = 10
SMOOTHING_THRESHOLD = 0.7 # Need >= 70% of buffer to be the same prediction

def initialize_resources():
    """Loads model and initializes MediaPipe. Called once at startup."""
    global clf, scaler, mp_hands, hands, prediction_buffer, stable_prediction_global
    print("Initializing resources...")

    # --- Load the Model ---
    try:
        with open(MODEL_PATH, 'rb') as f:
            # If your pickle file contains both classifier and scaler:
            # model_data = pickle.load(f)
            # clf = model_data['classifier']
            # scaler = model_data['scaler']
            # --- OR ---
            # If your pickle file *only* contains the classifier:
            clf = pickle.load(f)
            scaler = None # Explicitly set to None if no scaler saved
        print(f"Model loaded successfully from {MODEL_PATH}.")
        if scaler:
            print("Scaler loaded.")
        else:
            print("No scaler loaded (or not saved in pickle).")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
        clf = None
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        clf = None

    # --- Initialize MediaPipe ---
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,      # Process video stream
            max_num_hands=1,              # Detect only one hand
            min_detection_confidence=0.5, # Default confidence
            min_tracking_confidence=0.5   # Default confidence
        )
        print("MediaPipe Hands initialized.")
    except Exception as e:
        print(f"FATAL ERROR initializing MediaPipe: {e}")
        hands = None

    # --- Initialize Prediction Buffer ---
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    stable_prediction_global = "..." # Reset prediction after init

    print("Initialization complete.")

def process_frame(frame):
    """Processes a single frame to detect hands and predict signs."""
    global stable_prediction_global # Use and modify the global variable

    if not clf or not hands:
        # Draw error message if model or MediaPipe failed to load
        error_msg = "Error: Model or MediaPipe not loaded"
        cv2.putText(frame, error_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return frame # Return the original frame with error message

    # Flip the frame horizontally for a later selfie-view display
    # and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_prediction_display = "No hand detected"
    confidence_display = ""
    stable_prediction_display = stable_prediction_global # Use the latest stable one by default

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Get the first detected hand

        # Draw landmarks and connections
        mp.solutions.drawing_utils.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        try:
            # --- Feature Extraction (MUST MATCH TRAINING EXACTLY) ---
            landmarks = hand_landmarks.landmark
            # 1. Get all coordinates
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            # 2. Calculate relative coordinates to the wrist (landmark 0)
            wrist = coords[0]
            relative_coords = coords - wrist
            # 3. Normalize by the maximum distance from wrist to any *other* landmark
            #    Avoid division by zero if hand is just a dot.
            max_dist = np.max(np.linalg.norm(relative_coords[1:], axis=1))
            if max_dist < 1e-6: # Threshold to prevent division by zero/tiny values
                normalized_coords = np.zeros_like(relative_coords) # Or handle as an error/skip
            else:
                 normalized_coords = relative_coords / max_dist
            # 4. Flatten into a feature vector
            #    Exclude the wrist (landmark 0) if your model was trained that way,
            #    otherwise, include it. Assuming it was trained *with* wrist (relative to itself = 0,0,0)
            #    If trained WITHOUT wrist: feature_vector = normalized_coords[1:].flatten().reshape(1, -1)
            feature_vector = normalized_coords.flatten().reshape(1, -1) # Shape (1, 63)

            # --- Scaling (Optional, if you used a scaler during training) ---
            if scaler:
                 feature_vector_final = scaler.transform(feature_vector)
            else:
                 feature_vector_final = feature_vector # No scaling needed/applied

            # --- Prediction ---
            if hasattr(clf, 'predict_proba'):
                prediction_proba = clf.predict_proba(feature_vector_final)
                confidence = np.max(prediction_proba)
                predicted_class_index = np.argmax(prediction_proba)
                prediction = clf.classes_[predicted_class_index]
                current_prediction_display = prediction
                confidence_display = f"({confidence*100:.1f}%)"
            else: # Handle classifiers without predict_proba (like basic SVM)
                prediction = clf.predict(feature_vector_final)[0]
                current_prediction_display = prediction
                confidence_display = "(N/A)" # Confidence not available

            # --- Smoothing Logic ---
            prediction_buffer.append(prediction)
            # Only check for stable prediction if buffer is full
            if len(prediction_buffer) == PREDICTION_BUFFER_SIZE:
                counts = {pred: prediction_buffer.count(pred) for pred in set(prediction_buffer)}
                most_common_pred = max(counts, key=counts.get)
                # Check if the most common prediction meets the threshold
                if counts[most_common_pred] >= int(PREDICTION_BUFFER_SIZE * SMOOTHING_THRESHOLD):
                    if stable_prediction_global != most_common_pred:
                         print(f"New stable prediction: {most_common_pred}") # Log change
                         stable_prediction_global = most_common_pred # Update global state
                    stable_prediction_display = stable_prediction_global
                else:
                    # If threshold not met, keep displaying the *last* stable prediction
                    stable_prediction_display = stable_prediction_global
            else:
                 # Buffer not full yet, keep displaying last stable one
                 stable_prediction_display = stable_prediction_global

        except Exception as e:
            print(f"Error processing/predicting landmarks: {e}")
            current_prediction_display = "Processing Error"
            prediction_buffer.clear() # Clear buffer on error
            stable_prediction_display = stable_prediction_global # Keep last stable

    else: # No hand detected
         prediction_buffer.clear() # Clear buffer if no hand is detected
         stable_prediction_display = stable_prediction_global # Keep last stable

    # --- Draw Predictions on the image ---
    # Current (raw) prediction
    cv2.putText(image, f"Detect: {current_prediction_display} {confidence_display}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    # Stable (smoothed) prediction
    cv2.putText(image, f"Stable: {stable_prediction_display}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return image

def generate_frames():
    """Generator function to yield processed video frames for streaming."""
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0) # Use camera 0 (or change if needed)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        # Create a black image with error text
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Could not open camera", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        while True:
             ret, buffer = cv2.imencode('.jpg', error_img)
             if not ret: continue
             frame_bytes = buffer.tobytes()
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
             time.sleep(1) # Avoid busy-waiting

    print("Camera opened successfully. Starting frame generation.")
    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture frame. Skipping.")
                time.sleep(0.5) # Prevent tight loop on continuous error
                continue

            frame_count += 1
            # Process the frame using your logic
            # process_frame updates stable_prediction_global internally
            processed_frame = process_frame(frame)

            # Encode the processed frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # Adjust quality if needed
            if not ret:
                print(f"Error: Failed to encode frame {frame_count}. Skipping.")
                continue

            # Convert buffer to bytes
            frame_bytes = buffer.tobytes()

            # Yield the frame in MJPEG format for the video feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Optional: Add a small delay to control frame rate slightly
            # time.sleep(0.01) # ~100 fps theoretical max without processing load

    except Exception as e:
        print(f"Error during frame generation loop: {e}")
    finally:
        print("Releasing camera.")
        cap.release()
        if hands:
            # hands.close() # Usually not strictly necessary for Hands, but good practice for some solutions
            pass
        print("Frame generation stopped.")


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    print("Request received for index page.")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provides the video stream."""
    print("Request received for video feed.")
    # Returns a streaming response using the generator function
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """Provides the latest stable prediction as plain text."""
    # This route is polled by the JavaScript in index.html
    # It directly returns the global variable's current value
    return Response(stable_prediction_global, mimetype='text/plain')

# --- Main Execution ---
if __name__ == '__main__':
    initialize_resources() # Load model and MediaPipe *before* starting the server
    if clf and hands:
        print("Starting Flask server...")
        # Use host='0.0.0.0' to make accessible on your network
        # threaded=True is important for handling multiple requests (video + prediction polling)
        # use_reloader=False prevents Flask from restarting the process, which would
        # try to re-initialize the camera and global resources, often causing issues.
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    else:
        print("Application cannot start due to initialization errors.")