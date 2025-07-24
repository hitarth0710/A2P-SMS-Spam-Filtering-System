import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def load_test_messages(file_path, num_messages=100):
    """Load test messages from a CSV file."""
    try:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(file_path):
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to project root if needed
            project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
            # Create absolute path
            abs_file_path = os.path.join(project_root, file_path)
        else:
            abs_file_path = file_path

        # Check if file exists
        if not os.path.exists(abs_file_path):
            logging.error(f"Data file not found at {abs_file_path}")
            logging.info(f"Current working directory: {os.getcwd()}")
            logging.info(f"Checking if directory exists: {os.path.dirname(abs_file_path)}")
            if os.path.exists(os.path.dirname(abs_file_path)):
                logging.info(f"Directory exists, listing contents: {os.listdir(os.path.dirname(abs_file_path))}")
            return None

        logging.info(f"Loading test data from {abs_file_path}")
        df = pd.read_csv(abs_file_path)

        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Available columns: {df.columns.tolist()}")

        # Find message column
        message_col = None
        if 'cleaned_message' in df.columns:
            message_col = 'cleaned_message'
        elif 'Message' in df.columns:
            message_col = 'Message'
        elif 'message' in df.columns:
            message_col = 'message'

        if message_col is None:
            logging.error(f"Message column not found. Available: {df.columns.tolist()}")
            return None

        logging.info(f"Using '{message_col}' as message column")

        # Clean the data - remove any NaN or empty values
        df = df.dropna(subset=[message_col])
        df = df[df[message_col].str.strip() != '']

        # Sample random messages
        if len(df) < num_messages:
            logging.warning(f"Only {len(df)} messages available, using all of them")
            messages = df[message_col].tolist()
        else:
            sample_df = df.sample(num_messages)
            messages = sample_df[message_col].tolist()

        logging.info(f"Selected {len(messages)} messages for testing")
        return messages

    except Exception as e:
        logging.error(f"Error loading test messages: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def test_single_message(api_url, message):
    """Test a single message with the API and measure response time."""
    try:
        start_time = time.time()
        response = requests.post(
            api_url,
            headers={'Content-Type': 'application/json'},
            json={'message': message}
        )
        end_time = time.time()

        response_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()

            # Debug: Log the first few responses to see the structure
            if hasattr(test_single_message, 'debug_count'):
                test_single_message.debug_count += 1
            else:
                test_single_message.debug_count = 1

            if test_single_message.debug_count <= 3:
                logging.info(f"DEBUG - API Response {test_single_message.debug_count}: {result}")

            return {
                'status_code': response.status_code,
                'response_time': response_time,
                'is_spam': result.get('is_spam', None),
                'score': result.get('spam_probability', None),
                'success': True,
                'raw_response': result  # Keep raw response for debugging
            }
        else:
            return {
                'status_code': response.status_code,
                'response_time': response_time,
                'error': response.text,
                'success': False
            }
    except Exception as e:
        return {
            'status_code': 0,
            'response_time': 0,
            'error': str(e),
            'success': False
        }


def test_api_performance(api_url, messages, num_threads=4):
    """Test API performance with multiple messages in parallel."""
    results = []

    def process_message(args):
        idx, message = args
        result = test_single_message(api_url, message)
        result['message'] = message[:50] + '...' if len(message) > 50 else message
        result['message_id'] = idx + 1
        return result

    logging.info(f"Testing {len(messages)} messages with {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_message, enumerate(messages)))

    return results


def analyze_results(results, output_file=None):
    """Analyze and display API test results."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate success rate
    success_count = sum(df['success'])
    success_rate = success_count / len(df) * 100

    logging.info(f"\n{'=' * 50}")
    logging.info(f"API CONNECTION TEST RESULTS")
    logging.info(f"{'=' * 50}")
    logging.info(f"Total messages tested: {len(df)}")
    logging.info(f"Successful responses: {success_count}")
    logging.info(f"Failed responses: {len(df) - success_count}")
    logging.info(f"Success Rate: {success_rate:.2f}%")

    # Calculate response time statistics for successful requests
    successful_df = df[df['success'] == True]
    if not successful_df.empty:
        avg_response_time = successful_df['response_time'].mean()
        median_response_time = successful_df['response_time'].median()
        max_response_time = successful_df['response_time'].max()
        min_response_time = successful_df['response_time'].min()

        logging.info(f"\nRESPONSE TIME STATISTICS:")
        logging.info(f"Average: {avg_response_time:.4f} seconds")
        logging.info(f"Median: {median_response_time:.4f} seconds")
        logging.info(f"Minimum: {min_response_time:.4f} seconds")
        logging.info(f"Maximum: {max_response_time:.4f} seconds")

        # Debug: Show what fields are available in responses
        if 'raw_response' in successful_df.columns:
            sample_response = successful_df['raw_response'].iloc[0]
            logging.info(f"\nDEBUG - Sample API response structure: {sample_response}")
            if isinstance(sample_response, dict):
                logging.info(f"Available response fields: {list(sample_response.keys())}")

        # Show prediction distribution (handle None values)
        valid_predictions = successful_df['is_spam'].dropna()
        if not valid_predictions.empty:
            spam_predictions = sum(valid_predictions)
            non_spam_predictions = len(valid_predictions) - spam_predictions

            logging.info(f"\nPREDICTION DISTRIBUTION:")
            logging.info(f"Spam predictions: {spam_predictions}")
            logging.info(f"Non-spam predictions: {non_spam_predictions}")

            # Show invalid predictions if any
            invalid_predictions = len(successful_df) - len(valid_predictions)
            if invalid_predictions > 0:
                logging.info(f"Invalid/None predictions: {invalid_predictions}")
        else:
            logging.warning("No valid predictions found in successful responses")
            # Debug: Show what we're getting instead
            logging.info(f"DEBUG - is_spam values: {successful_df['is_spam'].unique()}")
            logging.info(f"DEBUG - score values: {successful_df['score'].unique()}")

    # Show errors if any
    failed_df = df[df['success'] == False]
    if not failed_df.empty:
        logging.info(f"\nERROR SUMMARY:")
        error_counts = failed_df['status_code'].value_counts()
        for status_code, count in error_counts.items():
            logging.info(f"Status code {status_code}: {count} occurrences")

    # Plot response time histogram
    if not successful_df.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(successful_df['response_time'], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('API Response Time Distribution')
        plt.grid(True, alpha=0.3)

        # Save or display the plot
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            logging.info(f"Response time histogram saved to {output_file}")
        else:
            plt.show()

    # Create simple report
    if output_file:
        report_file = output_file.replace('.png', '_report.txt')
        with open(report_file, 'w') as f:
            f.write("API Connection Test Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total messages tested: {len(df)}\n")
            f.write(f"Successful responses: {success_count}\n")
            f.write(f"Failed responses: {len(df) - success_count}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n\n")

            if not successful_df.empty:
                f.write("Response Time Statistics:\n")
                f.write(f"  Average: {avg_response_time:.4f} seconds\n")
                f.write(f"  Median: {median_response_time:.4f} seconds\n")
                f.write(f"  Minimum: {min_response_time:.4f} seconds\n")
                f.write(f"  Maximum: {max_response_time:.4f} seconds\n\n")

                valid_predictions = successful_df['is_spam'].dropna()
                if not valid_predictions.empty:
                    spam_predictions = sum(valid_predictions)
                    non_spam_predictions = len(valid_predictions) - spam_predictions
                    f.write("Prediction Distribution:\n")
                    f.write(f"  Spam predictions: {spam_predictions}\n")
                    f.write(f"  Non-spam predictions: {non_spam_predictions}\n\n")

            if not failed_df.empty:
                f.write("Errors:\n")
                for i, row in failed_df.iterrows():
                    f.write(f"  Message {row['message_id']}: Status {row['status_code']}\n")
                    if 'error' in row:
                        f.write(f"    Error: {row['error']}\n")

        logging.info(f"Detailed report saved to {report_file}")

    return {
        'total_messages': len(df),
        'successful_responses': success_count,
        'success_rate': success_rate,
        'avg_response_time': avg_response_time if not successful_df.empty else 0,
        'median_response_time': median_response_time if not successful_df.empty else 0,
        'max_response_time': max_response_time if not successful_df.empty else 0,
    }

def main():
    """Main function to test API connection and performance."""
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Configuration
    api_url = "http://localhost:8000/check_sms"
    data_path = "data/processed/labeled_messages.csv"
    num_messages = 1000
    output_file = "results/api_connection_test.png"

    logging.info("Starting API connection test...")

    # Check if the API is running
    try:
        response = requests.get("http://localhost:8000/")
        logging.info("✓ API is running and reachable")
    except requests.exceptions.ConnectionError:
        logging.error("✗ Cannot connect to API at http://localhost:8000/")
        logging.error("Make sure the API server is running first")
        return

    # Load test messages
    messages = load_test_messages(data_path, num_messages)
    if not messages:
        logging.error("Failed to load test messages")
        return

    logging.info(f"✓ Loaded {len(messages)} test messages")

    # Test API performance
    results = test_api_performance(api_url, messages)

    # Analyze results
    summary = analyze_results(results, output_file)

    logging.info(f"\n{'='*50}")
    logging.info("API connection test completed!")
    logging.info(f"{'='*50}")


if __name__ == "__main__":
    main()