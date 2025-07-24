import os
import argparse
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for the application"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )

def create_dirs():
    """Create necessary directories if they don't exist"""
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="A2P SMS Spam Filter")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--api', action='store_true', help='Start the API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API server')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host for the API server')
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    # Setup logging and create necessary directories
    setup_logging()
    create_dirs()
    
    # Parse command line arguments
    args = parse_args()
    
    if args.train:
        from src.models.train_model import train_model
        from src.data.preprocessing import preprocess_dataset
        
        # Preprocess the dataset
        logging.info("Preprocessing dataset...")
        preprocess_dataset(
            input_path="data/raw/message_dataset_50k.csv",
            output_path="data/processed/labeled_messages.csv"
        )
        
        # Train the model
        logging.info("Training model...")
        metrics = train_model(
            data_path="data/processed/labeled_messages.csv",
            model_output_path="data/models/model.pkl"
        )
        
        logging.info(f"Model training completed with accuracy: {metrics['accuracy']:.4f}")
    
    if args.api:
        # Import the Flask app
        from api.app import app
        
        # Start the Flask server
        logging.info(f"Starting Flask API server on {args.host}:{args.port}...")
        app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()