
const predictionDisplayElement = document.getElementById('detected-sign-display');
const videoElement = document.getElementById('video-feed');
const instructionTextElement = document.getElementById('instruction-text');
const targetImageElement = document.getElementById('target-image');
const feedbackElement = document.getElementById('feedback');
const signButtons = document.querySelectorAll('#sign-buttons button'); // Get all practice buttons

// --- State Variables ---
let currentTargetSign = null; // The sign the user is currently trying to make (e.g., "A")
let stablePrediction = "..."; // Store the latest stable prediction received from backend
let correctSignStartTime = null; // Timestamp (ms) when correct sign was first detected stably
const REQUIRED_HOLD_TIME_MS = 2000; // Hold for 2 seconds (Adjust as needed)
const POLLING_INTERVAL_MS = 250; // Check prediction from backend every 250ms (Adjust as needed)

let pollingTimer = null; // Holds the interval timer ID

// --- Practice Control ---
function startPractice(sign) {
    console.log("Starting practice for:", sign);
    currentTargetSign = sign;
    correctSignStartTime = null; // Reset hold timer

    // --- Update UI Elements ---
    if (instructionTextElement) instructionTextElement.textContent = `Now, make the sign for: ${sign}`;

    // Use the global 'staticBaseUrl' passed from the HTML template
    if (targetImageElement && typeof staticBaseUrl !== 'undefined') {
        targetImageElement.src = staticBaseUrl + `images/${sign.toUpperCase()}.png`; // Added 'images/' prefix
        targetImageElement.alt = `Example sign for ${sign}`;
        targetImageElement.style.display = 'block'; // Show the image
    } else if (targetImageElement) {
         console.error("staticBaseUrl is not defined. Cannot set image source.");
         targetImageElement.style.display = 'none';
    }


    if (feedbackElement) {
        feedbackElement.textContent = 'Hold the sign...';
        feedbackElement.className = ''; // Reset feedback color/style
    }


    // Highlight the active button and deactivate others
    signButtons.forEach(button => {
        if (button.textContent === sign) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });

    // --- Start/Ensure Polling ---
    // Clear any existing timer before starting a new one
    if (pollingTimer) {
        clearInterval(pollingTimer);
    }
    // Start polling the backend for predictions, only if URL is defined
    if (typeof predictionUrl !== 'undefined') {
        pollingTimer = setInterval(fetchPredictionAndCheck, POLLING_INTERVAL_MS);
        // Fetch immediately once to get the initial state after selection
        fetchPredictionAndCheck();
    } else {
        console.error("predictionUrl is not defined. Cannot start polling.");
        if(feedbackElement) feedbackElement.textContent = "Setup Error: Missing URL";
        if(feedbackElement) feedbackElement.className = 'incorrect';
    }
}

// --- Prediction Fetching and Checking ---
async function fetchPredictionAndCheck() {
    // Only proceed if we are actively practicing a sign and URL is known
    if (!currentTargetSign || typeof predictionUrl === 'undefined') {
        // console.log("Polling skipped, no target sign or missing URL.");
        return;
    }

    try {
        // Use the global 'predictionUrl' variable passed from the HTML template
        const response = await fetch(predictionUrl); // Use variable
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const predictionText = await response.text();
        stablePrediction = predictionText || "..."; // Update global JS variable
        if (predictionDisplayElement) predictionDisplayElement.textContent = stablePrediction; // Update the debug display

        updateFeedback(); // Check if the detected sign matches the target

    } catch (error) {
        console.error('Error fetching prediction:', error);
        stablePrediction = "Conn Error"; // Indicate a connection error
        if (predictionDisplayElement) predictionDisplayElement.textContent = stablePrediction;
        if (feedbackElement) {
            feedbackElement.textContent = 'Connection Error';
            feedbackElement.className = 'incorrect';
        }

        // Stop polling on error
        if (pollingTimer) {
            clearInterval(pollingTimer);
            pollingTimer = null;
        }
    }
}

// --- Feedback Logic ---
function updateFeedback() {
    // Should only update feedback if a practice target is set and element exists
    if (!feedbackElement) return; // Exit if feedback element not found

    if (!currentTargetSign) {
        feedbackElement.textContent = 'Select a sign to practice.';
        feedbackElement.className = '';
        return;
    }

    // Compare detected sign with target sign (case-insensitive)
    const currentStableUpper = stablePrediction.toUpperCase();
    const targetUpper = currentTargetSign.toUpperCase();

    if (currentStableUpper === targetUpper) {
        // --- CORRECT SIGN DETECTED ---
        if (correctSignStartTime === null) {
            // Correct sign detected for the first time (stably), start the timer
            correctSignStartTime = Date.now();
            feedbackElement.textContent = 'Keep holding...';
            feedbackElement.className = 'holding'; // Yellow/Orange color
        } else {
            // Correct sign is still being held, check elapsed time
            const elapsedTime = Date.now() - correctSignStartTime;
            if (elapsedTime >= REQUIRED_HOLD_TIME_MS) {
                // --- SUCCESS: Held long enough! ---
                feedbackElement.textContent = `Correct! You held '${currentTargetSign}'!`;
                feedbackElement.className = 'success'; // Green color

                // Stop checking for this sign (practice complete for this round)
                currentTargetSign = null;
                correctSignStartTime = null;

                // Stop polling
                if (pollingTimer) {
                    clearInterval(pollingTimer);
                    pollingTimer = null;
                }

                // Remove active class from button after success
                signButtons.forEach(button => button.classList.remove('active'));

                // Optionally, update instruction text after success
                if (instructionTextElement) instructionTextElement.textContent = "Well done! Select another sign.";
                // Hide target image after success
                if (targetImageElement) targetImageElement.style.display = 'none';

            } else {
                // --- STILL HOLDING CORRECTLY (Timer running) ---
                // Update feedback with remaining time (optional but helpful)
                const remaining = ((REQUIRED_HOLD_TIME_MS - elapsedTime) / 1000).toFixed(1);
                feedbackElement.textContent = `Keep holding... (${remaining}s)`;
                feedbackElement.className = 'holding'; // Yellow/Orange color
            }
        }
    } else {
        // --- INCORRECT SIGN or NO/ERROR DETECTION ---
        correctSignStartTime = null; // Reset timer if sign is wrong or lost

        if (stablePrediction === "..." || stablePrediction === "No hand detected" || stablePrediction === "Initializing..." || stablePrediction === "System Error") {
            // Waiting for a hand or stable prediction or system issue reported by backend
            feedbackElement.textContent = `Show the '${currentTargetSign}' sign... (Status: ${stablePrediction})`;
            feedbackElement.className = ''; // Default color
        } else if (stablePrediction === "Conn Error" || stablePrediction === "Processing Error") {
            // Show specific JS/backend errors distinctly
            feedbackElement.textContent = `System Status: ${stablePrediction}`;
            feedbackElement.className = 'incorrect'; // Red color
        }
        else {
            // A specific, but incorrect, sign is detected
            feedbackElement.textContent = `Detected: ${stablePrediction}. Try '${currentTargetSign}'!`;
            feedbackElement.className = 'incorrect'; // Red color
        }
    }
}

// --- Video Stream Event Handlers ---
// Make sure videoElement exists before adding listeners
if (videoElement) {
    videoElement.onerror = () => {
        console.error("Video stream failed to load.");
        // Try to provide feedback in the practice area if possible
        if (instructionTextElement) instructionTextElement.textContent = "Camera Error!";
        if (targetImageElement) targetImageElement.style.display = 'none'; // Hide image placeholder
        if (feedbackElement) {
            feedbackElement.textContent = "Cannot load camera feed.";
            feedbackElement.className = 'incorrect'; // Red color
        }
        if (predictionDisplayElement) predictionDisplayElement.textContent = "N/A";
        // Stop polling if video fails
        if (pollingTimer) {
            clearInterval(pollingTimer);
            pollingTimer = null;
            console.log("Prediction polling stopped due to video error.");
        }
        // Optionally display an error message directly in the video container
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.innerHTML = '<p style="color: red; text-align: center; padding: 20px;">Error loading video stream. Check camera permissions and connection.</p>';
        }
    };

    videoElement.onload = () => {
        console.log("Video stream started successfully.");
        // Fetch initial prediction state for the debug display when video loads, if URL is known
        if (typeof predictionUrl !== 'undefined') {
            fetch(predictionUrl) // Use variable
                .then(r => r.ok ? r.text() : Promise.reject(`HTTP ${r.status}`))
                .then(text => {
                    if (predictionDisplayElement) predictionDisplayElement.textContent = text || "...";
                })
                .catch(err => {
                    console.error("Initial prediction fetch failed:", err);
                    if (predictionDisplayElement) predictionDisplayElement.textContent = "Error";
                });
        } else {
            console.error("predictionUrl is not defined. Cannot fetch initial prediction.");
             if (predictionDisplayElement) predictionDisplayElement.textContent = "Setup Error";
        }
    };
} else {
    console.error("Video feed element (#video-feed) not found.");
     // Provide some feedback if possible
     if(feedbackElement) feedbackElement.textContent = "Error: Video player missing.";
     if(feedbackElement) feedbackElement.className = 'incorrect';
}


// Ensure the script runs after the DOM is loaded by using the `defer` attribute
// in the HTML script tag. No need for DOMContentLoaded listener here.