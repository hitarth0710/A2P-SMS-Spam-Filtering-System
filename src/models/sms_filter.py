# src/models/sms_filter.py
import joblib
import logging
import os
from src.utils.whitelist import WhitelistManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sms_filter.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("sms_filter")


class SMSFilterService:
    """Service to filter SMS messages using ML and whitelist"""

    def __init__(self, model_path, whitelist_config):
        """
        Initialize the SMS filter service

        Args:
            model_path: Path to the trained classifier model
            whitelist_config: Path to the whitelist configuration
        """
        self.model_path = model_path
        self.whitelist_config = whitelist_config
        self.model = None
        self.whitelist = None
        self._load_resources()

    def _load_resources(self):
        """Load the model and whitelist configuration"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)

            logger.info(f"Loading whitelist from {self.whitelist_config}")
            self.whitelist = WhitelistManager(self.whitelist_config)
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            raise

    def check_message(self, message, sender_id=None):
        """
        Check if a message should be allowed or blocked

        Args:
            message: The SMS message to check
            sender_id: Optional sender ID

        Returns:
            dict: Result with verdict and reason
        """
        # Log the incoming message
        logger.info(f"Checking message: {message[:50]}...")

        # First check whitelist
        if self.whitelist.is_whitelisted(message, sender_id):
            logger.info("Message whitelisted")
            return {
                "verdict": "allowed",
                "reason": "whitelisted",
                "category": None
            }

        # If not whitelisted, apply the ML model
        prediction = self.model.predict([message])[0]

        # Check if prediction is spam
        if prediction == "spam":
            logger.info(f"Message blocked: classified as {prediction}")
            return {
                "verdict": "blocked",
                "reason": "ai",
                "category": prediction
            }
        else:
            logger.info(f"Message allowed: classified as {prediction}")
            return {
                "verdict": "allowed",
                "reason": "ai",
                "category": prediction
            }

    def get_confidence_scores(self, message):
        """
        Get confidence scores for each category

        Args:
            message: The SMS message to check

        Returns:
            dict: Confidence scores for each category
        """
        # Get probability scores from model
        proba = self.model.predict_proba([message])[0]
        classes = self.model.classes_

        # Create a dictionary of category: probability
        confidence = {class_name: float(prob) for class_name, prob in zip(classes, proba)}

        return confidence