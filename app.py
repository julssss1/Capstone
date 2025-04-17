import warnings
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque
import time
from flask import Flask, render_template, Response, url_for # Added url_for

# Suppress specific UserWarning from sklearn about feature names
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but")

# --- Flask App Initialization ---
# Flask automatically looks for 'templates' and 'static' folders
# Ensure 'static' folder exists and contains Student.css and sign images (A.png, B.png etc.)
app = Flask(__name__)

# --- Global Variables ---
MODEL_PATH = 'svm_sign_model_normalized.pkl' # Ensure this path is correct
clf = None
scaler = None # Set to None if your .pkl file only contains the classifier
mp_hands = None
hands = None
prediction_buffer = None
stable_prediction_global = "Initializing..." # Initial state

# --- Prediction Smoothing Parameters ---
PREDICTION_BUFFER_SIZE = 10 # Number of frames to consider for stability
SMOOTHING_THRESHOLD = 0.7 # Need >= 70% of buffer to be the same prediction for stability

def initialize_resources():
    """Loads model and initializes MediaPipe. Called once at startup."""
    global clf, scaler, mp_hands, hands, prediction_buffer, stable_prediction_global
    print("Initializing resources...")

    # --- Load the Model ---
    try:
        with open(MODEL_PATH, 'rb') as f:
            # --- OPTION 1: If your pickle file *only* contains the classifier: ---
            clf = pickle.load(f)
            scaler = None # Explicitly set to None if no scaler saved
            print(f"Model (classifier only) loaded successfully from {MODEL_PATH}.")

            # --- OPTION 2: If your pickle file contains both classifier and scaler (e.g., in a dictionary): ---
            # try:
            #     model_data = pickle.load(f)
            #     clf = model_data['classifier']
            #     scaler = model_data['scaler']
            #     print(f"Model (classifier and scaler) loaded successfully from {MODEL_PATH}.")
            # except (KeyError, TypeError):
            #     # Fallback if structure is different or only classifier is present
            #     f.seek(0) # Rewind file pointer
            #     clf = pickle.load(f)
            #     scaler = None
            #     print(f"Model (fallback: classifier only) loaded successfully from {MODEL_PATH}.")

        if scaler:
            print("Scaler loaded.")
        else:
            print("No scaler loaded (or not saved in pickle). Ensure features match training.")
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
            min_detection_confidence=0.5, # Adjust as needed
            min_tracking_confidence=0.5   # Adjust as needed
        )
        mp_drawing = mp.solutions.drawing_utils # Store drawing utils
        mp_drawing_styles = mp.solutions.drawing_styles # Store drawing styles
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
        error_msg = "Error: Model or MediaPipe not loaded"
        # Increase font size and thickness for visibility on video feed
        cv2.putText(frame, error_msg, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        stable_prediction_global = "System Error" # Update global state on critical failure
        return frame

    # Process the frame
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB) # Flip and convert BGR -> RGB
    image.flags.writeable = False # Performance optimization
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back RGB -> BGR for display

    current_prediction_display = "No hand detected"
    confidence_display = ""
    # Start with the current stable prediction for display this frame
    stable_prediction_display = stable_prediction_global

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Get the first detected hand

        # Draw landmarks on the image using stored utils/styles
        mp.solutions.drawing_utils.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        try:
            # --- Feature Extraction (MUST MATCH TRAINING) ---
            landmarks = hand_landmarks.landmark
            # 1. Get all landmark coordinates (x, y, z)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            # 2. Make coordinates relative to the wrist (landmark 0)
            wrist = coords[0]
            relative_coords = coords - wrist
             # 3. Normalize by the maximum distance from wrist to any other landmark
            max_dist = np.max(np.linalg.norm(relative_coords[1:], axis=1)) # Avoid dividing by zero if hand collapsed
            if max_dist < 1e-6: # Check for very small max_dist (potential issue)
                 # Handle case where hand landmarks are too close (e.g., clenched fist near wrist)
                 # Add small epsilon to avoid division by zero but signal potential issue
                 normalized_coords = relative_coords / (max_dist + 1e-6)
            else:
                 normalized_coords = relative_coords / max_dist

            # 4. Flatten into a feature vector (shape depends on whether wrist is included)
            # !! IMPORTANT: Choose the flatten method that matches your training !!
            # Example 1: Trained WITH wrist (landmark 0) included (resulting in 21*3 = 63 features)
            feature_vector = normalized_coords.flatten().reshape(1, -1) # Shape (1, 63)
            # Example 2: Trained WITHOUT wrist (using landmarks 1-20) (resulting in 20*3 = 60 features)
            # feature_vector = normalized_coords[1:].flatten().reshape(1, -1) # Shape (1, 60)
            # -------------------------------------------------------------

            # 5. Apply scaler if it was loaded and used during training
            if scaler:
                 feature_vector_final = scaler.transform(feature_vector)
            else:
                 feature_vector_final = feature_vector # Use unscaled features

            # --- Prediction ---
            prediction = ""
            if hasattr(clf, 'predict_proba'): # Check if the classifier supports probability estimates
                prediction_proba = clf.predict_proba(feature_vector_final)
                confidence = np.max(prediction_proba)
                # Only assign prediction if confidence meets a threshold (e.g., 50%)
                pred_idx = np.argmax(prediction_proba)
                if confidence > 0.5: # Adjust threshold as needed
                    prediction = clf.classes_[pred_idx]
                    current_prediction_display = str(prediction) # Ensure it's a string
                    confidence_display = f"({confidence*100:.1f}%)"
                else:
                    prediction = "Unknown" # Assign if low confidence
                    current_prediction_display = "Unknown"
                    confidence_display = f"({confidence*100:.1f}%)"

            else: # Fallback for classifiers without predict_proba (like basic SVM)
                prediction = clf.predict(feature_vector_final)[0]
                current_prediction_display = str(prediction) # Ensure it's a string
                confidence_display = "(N/A)"

            # --- Smoothing Logic ---
            prediction_buffer.append(prediction) # Add current prediction (could be 'Unknown') to the buffer

            # Only check for stability once the buffer is full
            if len(prediction_buffer) == PREDICTION_BUFFER_SIZE:
                # Count occurrences of each prediction in the buffer
                counts = {pred: prediction_buffer.count(pred) for pred in set(prediction_buffer)}
                # Find the prediction that occurred most often
                most_common_pred = max(counts, key=counts.get)

                # Calculate the minimum count needed to meet the threshold
                required_count = int(PREDICTION_BUFFER_SIZE * SMOOTHING_THRESHOLD)

                # Check if the count of the most common prediction meets the threshold
                # AND ensure the most common prediction is not 'Unknown' or error states
                if counts[most_common_pred] >= required_count and most_common_pred not in ["Unknown", "Processing Error", "System Error"]:
                    # Only update the global stable prediction if it has actually changed
                    if stable_prediction_global != most_common_pred:
                         # print(f"New stable prediction: {most_common_pred} (Count: {counts[most_common_pred]}/{PREDICTION_BUFFER_SIZE})") # Optional debug log
                         stable_prediction_global = most_common_pred
                    # Update the display variable for this frame to the confirmed stable prediction
                    stable_prediction_display = stable_prediction_global
                else:
                    # If threshold NOT met OR most common is 'Unknown'/error,
                    # keep displaying the LAST known VALID stable prediction.
                    # stable_prediction_global remains unchanged UNLESS it was an error state before.
                    # Resetting to "..." if previous was error might be better?
                    if stable_prediction_global in ["Processing Error", "System Error"]:
                         stable_prediction_global = "..." # Reset if previous was error
                    stable_prediction_display = stable_prediction_global
            else:
                 # If buffer is not full yet, keep displaying the last known stable prediction
                 stable_prediction_display = stable_prediction_global

        except Exception as e:
            print(f"Error processing/predicting landmarks: {e}")
            current_prediction_display = "Processing Error"
            prediction_buffer.clear() # Clear buffer on error
            stable_prediction_global = "Processing Error" # Report error state
            stable_prediction_display = stable_prediction_global

    else: # No hand detected in the frame
         current_prediction_display = "No hand detected"
         prediction_buffer.clear() # Clear buffer when no hand is seen
         # Keep displaying the last known stable prediction even if hand disappears briefly
         # Reset if previous state was an error
         if stable_prediction_global in ["Processing Error", "System Error"]:
             stable_prediction_global = "..."
         stable_prediction_display = stable_prediction_global

    # --- Draw Predictions on the image ---
    # Use larger fonts and thicker lines for better visibility
    # Display the instantaneous prediction + confidence (Top Left - Blue)
    cv2.putText(image, f"Detect: {current_prediction_display} {confidence_display}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2, cv2.LINE_AA) # Blue for instant
    # Display the stable prediction (Below Instant - Green)
    cv2.putText(image, f"Stable: {stable_prediction_display}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA) # Green for stable

    return image

def generate_frames():
    """Generator function to yield processed video frames for streaming."""
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0) # Try default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        # Create a black image with an error message
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Could not open camera", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        while True:
             try:
                 ret, buffer = cv2.imencode('.jpg', error_img)
                 if not ret: continue
                 frame_bytes = buffer.tobytes()
                 yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                 time.sleep(1) # Yield error frame periodically
             except Exception as e:
                 print(f"Error yielding error frame: {e}")
                 break # Exit if even yielding the error fails


    print("Camera opened successfully. Starting frame generation.")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to capture frame. Skipping.")
                time.sleep(0.1) # Wait a bit before retrying
                continue

            # Process the frame to get predictions and draw landmarks/text
            processed_frame = process_frame(frame)

            # Encode the processed frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # Slightly higher quality
            if not ret:
                print(f"Warning: Failed to encode frame. Skipping.")
                continue

            # Convert buffer to bytes
            frame_bytes = buffer.tobytes()

            # Yield the frame in the multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # time.sleep(0.01) # Optional small delay

    except Exception as e:
        print(f"Error during frame generation loop: {e}")
    finally:
        print("Releasing camera.")
        cap.release()
        # Clean up MediaPipe resources (optional, might help on exit)
        if hands:
             try:
                hands.close()
             except Exception as e:
                 print(f"Error closing MediaPipe hands: {e}")
        print("Frame generation stopped.")


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main Student Dashboard page."""
    print("Request received for Student Dashboard page.")
    # Define the signs available for practice. These should match the filenames
    # in your static/ folder (e.g., A.png, B.png)
    available_signs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J","K", "L", "M", "N","O","P","Q","R","S","T","U", "V", "W","X", "Y","Z"] # Customize this list!
    # Render the StudentDashboard template, passing the available signs
    return render_template('StudentDashboard.html', available_signs=available_signs)

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
    # This route is polled by the JavaScript in StudentDashboard.html
    # It returns the current value of the global variable
    return Response(stable_prediction_global, mimetype='text/plain')

# --- Main Execution ---
if __name__ == '__main__':
    initialize_resources() # Load model and init MediaPipe before starting server
    if clf and hands: # Only start if resources loaded successfully
        print("Initialization successful. Starting Flask server...")
        # Run the Flask app
        # host='0.0.0.0' makes it accessible on your network
        # threaded=True allows handling multiple requests (feed + prediction polling)
        # use_reloader=False is often recommended when using hardware like cameras
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    else:
        print("Application cannot start due to initialization errors. Check model path and MediaPipe setup.")