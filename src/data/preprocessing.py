# src/data/preprocessing.py
import re
import pandas as pd


def clean_message(message):
    """
    Clean the message by lowercasing, removing special characters, and extra whitespace.

    Args:
        message (str): The raw message text

    Returns:
        str: The cleaned message
    """
    message_lower = message.lower()
    message_cleaned = re.sub(r'[^a-z0-9\s]', '', message_lower)
    message_cleaned = re.sub(r'\s+', ' ', message_cleaned).strip()
    return message_cleaned


def preprocess_dataset(input_path, output_path=None):
    """
    Preprocess the SMS dataset by removing duplicates and cleaning messages.

    Args:
        input_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the processed dataset

    Returns:
        pd.DataFrame: The processed DataFrame
    """
    # Load the dataset
    df = pd.read_csv(input_path)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Clean messages
    df['cleaned_message'] = df['Message'].apply(clean_message)

    # Save to file if output_path is provided
    if output_path:
        df[['Message', 'Category', 'cleaned_message']].to_csv(output_path, index=False)

    return df
