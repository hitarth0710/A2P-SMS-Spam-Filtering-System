import logging
import joblib
from flask import Flask, request, jsonify
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import clean_message
from src.utils.whitelist import is_whitelisted, load_whitelist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/filter.log"),
        logging.StreamHandler()
    ]
)

# Initialize global variables
model = None
whitelisted_domains = []
whitelisted_phrases = []

# Load model and configuration
def load_model_and_config():
    """Load the model and whitelist configuration."""
    global model, whitelisted_domains, whitelisted_phrases

    try:
        # Load the trained model
        model = joblib.load("data/models/model.pkl")
        logging.info("Model loaded successfully")

        # Load whitelist configuration
        whitelisted_domains, whitelisted_phrases = load_whitelist("config/config.yaml")
        logging.info(f"Loaded {len(whitelisted_domains)} domains and {len(whitelisted_phrases)} phrases to whitelist")
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        # Continue without crashing, but services might be limited

# Initialize Flask app
app = Flask(__name__)

# Call the function to load model and config at startup
load_model_and_config()

@app.route("/check_sms", methods=["POST"])
def check_sms():
    """
    Check if an SMS message is spam, promotional, or transactional.

    Expected JSON:
    {
        "message": "Your SMS message to check"
    }
    """
    # Get JSON data from request
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Invalid request. 'message' field is required"}), 400

    message = data["message"]
    logging.info(f"Received message: '{message}'")

    try:
        # Check if model is loaded
        if model is None:
            load_model_and_config()
            if model is None:
                return jsonify({"error": "Model not loaded. Please try again later."}), 500

        # Check whitelist first
        if is_whitelisted(message, whitelisted_domains, whitelisted_phrases):
            logging.info(f"Message whitelisted: '{message}'")
            return jsonify({"verdict": "allowed", "reason": "whitelisted"})

        # Clean the message
        cleaned_message = clean_message(message)

        # Predict using the model
        prediction = model.predict([cleaned_message])[0]

        # Determine verdict based on prediction
        if prediction == "Spam":
            logging.info(f"Message blocked (Spam): '{message}'")
            return jsonify({"verdict": "blocked", "reason": "ai"})
        else:  # Transactional or Promotional
            logging.info(f"Message allowed ({prediction}): '{message}'")
            return jsonify({"verdict": "allowed", "reason": "ai"})

    except Exception as e:
        logging.error(f"Error processing message: {e}")
        return jsonify({"error": "Error processing message", "details": str(e)}), 500

# Add a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "healthy", "message": "A2P SMS Spam Filter API is running"})

# Add a root endpoint with documentation
@app.route("/", methods=["GET"])
def home():
    """Root endpoint with API documentation."""
    return jsonify({
        "name": "A2P SMS Spam Filter",
        "description": "API for classifying SMS messages as Transactional, Promotional, or Spam",
        "version": "1.0.0",
        "endpoints": {
            "/check_sms": {
                "method": "POST",
                "description": "Check if an SMS message is spam",
                "request_body": {"message": "Your SMS message to check"},
                "response": {"verdict": "allowed|blocked", "reason": "whitelisted|ai"}
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)