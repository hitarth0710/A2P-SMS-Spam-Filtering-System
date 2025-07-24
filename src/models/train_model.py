# src/models/train_model.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_model(data_path, model_output_path):
    """
    Train a message classifier using TF-IDF and Multinomial Naive Bayes.

    Args:
        data_path (str): Path to the labeled dataset CSV
        model_output_path (str): Path to save the trained model

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load the labeled dataset
    df = pd.read_csv(data_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_message'], df['Category'], test_size=0.25, random_state=42
    )

    # Create a pipeline for TF-IDF vectorization and Multinomial Naive Bayes classification
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Trained model saved to {model_output_path}")

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred).tolist()
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": confusion
    }
