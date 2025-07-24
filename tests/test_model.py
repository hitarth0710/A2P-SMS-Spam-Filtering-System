import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_model(model_path):
    """Load the trained model from the specified path."""
    # Convert to absolute path if it's a relative path
    if not os.path.isabs(model_path):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root if needed
        project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
        # Create absolute path
        abs_model_path = os.path.join(project_root, model_path)
    else:
        abs_model_path = model_path
    
    # Check if file exists
    if not os.path.exists(abs_model_path):
        logging.error(f"Model file not found at {abs_model_path}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Checking if directory exists: {os.path.dirname(abs_model_path)}")
        if os.path.exists(os.path.dirname(abs_model_path)):
            logging.info(f"Directory exists, listing contents: {os.listdir(os.path.dirname(abs_model_path))}")
        sys.exit(1)
        
    try:
        logging.info(f"Loading model from {abs_model_path}")
        model = joblib.load(abs_model_path)
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)


def load_data(data_path, test_size=0.2, random_state=42):
    """Load and split dataset for testing."""
    try:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(data_path):
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to project root if needed
            project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
            # Create absolute path
            abs_data_path = os.path.join(project_root, data_path)
        else:
            abs_data_path = data_path

        # Check if file exists
        if not os.path.exists(abs_data_path):
            logging.error(f"Data file not found at {abs_data_path}")
            logging.info(f"Current working directory: {os.getcwd()}")
            logging.info(f"Checking if directory exists: {os.path.dirname(abs_data_path)}")
            if os.path.exists(os.path.dirname(abs_data_path)):
                logging.info(f"Directory exists, listing contents: {os.listdir(os.path.dirname(abs_data_path))}")
            sys.exit(1)

        logging.info(f"Loading data from {abs_data_path}")
        df = pd.read_csv(abs_data_path)

        # Log dataset info
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Available columns: {df.columns.tolist()}")

        # Map actual column names to expected ones
        message_col = None
        label_col = None

        # Check for message column variations
        if 'cleaned_message' in df.columns:
            message_col = 'cleaned_message'  # Use cleaned version if available
        elif 'Message' in df.columns:
            message_col = 'Message'
        elif 'message' in df.columns:
            message_col = 'message'

        # Check for label column variations
        if 'Category' in df.columns:
            label_col = 'Category'
        elif 'label' in df.columns:
            label_col = 'label'

        if message_col is None or label_col is None:
            logging.error(f"Could not find required columns. Available: {df.columns.tolist()}")
            sys.exit(1)

        logging.info(f"Using '{message_col}' as message column and '{label_col}' as label column")

        # Clean the data - remove any NaN or empty values
        df = df.dropna(subset=[message_col, label_col])
        df = df[df[message_col].str.strip() != '']

        # Check unique labels and convert to binary if needed
        unique_labels = df[label_col].unique()
        logging.info(f"Unique labels: {unique_labels}")
        logging.info(f"Label types: {[type(label) for label in unique_labels]}")

        # Convert all labels to string first, then to binary integers
        df[label_col] = df[label_col].astype(str).str.lower().str.strip()
        unique_labels_clean = df[label_col].unique()
        logging.info(f"Cleaned unique labels: {unique_labels_clean}")

        # Convert labels to binary integers (0 for non-spam, 1 for spam)
        labels = df[label_col].apply(lambda x: 1 if x == 'spam' else 0)
        labels = labels.astype(int)  # Ensure integer type

        # Show label distribution
        logging.info(f"Label distribution: {labels.value_counts().to_dict()}")
        logging.info(f"Label data type: {labels.dtype}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[message_col],
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        # Ensure test labels are integers
        y_test = y_test.astype(int)

        logging.info(f"Data loaded successfully. Test set size: {len(X_test)}")
        logging.info(f"Label distribution in test set: {y_test.value_counts().to_dict()}")
        logging.info(f"Test labels data type: {y_test.dtype}")

        return X_test, y_test

    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data."""
    try:
        logging.info("Predicting on test data...")
        y_pred = model.predict(X_test)

        # Log prediction info for debugging
        logging.info(f"Raw predictions type: {type(y_pred)}")
        logging.info(f"Raw predictions unique values: {np.unique(y_pred)}")

        # Convert predictions to binary if they're categorical
        if isinstance(y_pred[0], str) or not np.issubdtype(y_pred.dtype, np.integer):
            # Convert string predictions to binary
            y_pred_binary = []
            for pred in y_pred:
                pred_str = str(pred).lower().strip()
                if pred_str == 'spam':
                    y_pred_binary.append(1)
                else:
                    y_pred_binary.append(0)  # All non-spam categories become 0
            y_pred = np.array(y_pred_binary, dtype=int)
        else:
            y_pred = np.array(y_pred, dtype=int)

        # Ensure y_test is integer
        y_test = np.array(y_test, dtype=int)

        logging.info(f"y_test type: {y_test.dtype}, unique values: {np.unique(y_test)}")
        logging.info(f"y_pred type: {y_pred.dtype}, unique values: {np.unique(y_pred)}")

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        # Print performance metrics
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # Generate and print classification report
        report = classification_report(y_test, y_pred)
        logging.info(f"Classification Report:\n{report}")

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate ROC curve and AUC if the model supports predict_proba
        try:
            y_prob = model.predict_proba(X_test)
            # Find the column that corresponds to spam (class 1)
            if hasattr(model, 'classes_'):
                spam_class_idx = None
                for i, cls in enumerate(model.classes_):
                    if str(cls).lower() == 'spam':
                        spam_class_idx = i
                        break

                if spam_class_idx is not None:
                    y_prob_spam = y_prob[:, spam_class_idx]
                else:
                    # If no 'spam' class found, assume last column is positive class
                    y_prob_spam = y_prob[:, -1]
            else:
                y_prob_spam = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]

            fpr, tpr, _ = roc_curve(y_test, y_prob_spam)
            roc_auc = auc(fpr, tpr)
            logging.info(f"AUC: {roc_auc:.4f}")
        except Exception as e:
            logging.warning(f"Model doesn't support probability predictions or error occurred: {e}")
            fpr, tpr, roc_auc = None, None, None

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'report': report,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'y_pred': y_pred,
            'y_test': y_test
        }

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
def plot_confusion_matrix(cm, output_path=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to {output_path}")

    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, output_path=None):
    """Plot and save ROC curve."""
    if fpr is None or tpr is None or roc_auc is None:
        logging.warning("ROC curve data not available, skipping plot")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f"ROC curve saved to {output_path}")

    plt.show()

def analyze_errors(X_test, y_test, y_pred):
    """Analyze prediction errors to understand model weaknesses."""
    # Convert to pandas Series if they're not already
    if not isinstance(X_test, pd.Series):
        X_test = pd.Series(X_test)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)

    # Find false positives (actual: not spam, predicted: spam)
    false_positives = X_test[(y_test == 0) & (y_pred == 1)]

    # Find false negatives (actual: spam, predicted: not spam)
    false_negatives = X_test[(y_test == 1) & (y_pred == 0)]

    logging.info(f"Number of false positives (legitimate messages classified as spam): {len(false_positives)}")
    logging.info(f"Number of false negatives (spam messages classified as legitimate): {len(false_negatives)}")

    # Display some examples of each error type
    if len(false_positives) > 0:
        logging.info("\nExamples of false positives (legitimate messages classified as spam):")
        for i, msg in enumerate(false_positives.sample(min(5, len(false_positives))).values):
            logging.info(f"{i+1}. {msg}")

    if len(false_negatives) > 0:
        logging.info("\nExamples of false negatives (spam messages classified as legitimate):")
        for i, msg in enumerate(false_negatives.sample(min(5, len(false_negatives))).values):
            logging.info(f"{i+1}. {msg}")

    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def analyze_model_bias(y_test, y_pred):
    """Analyze if the model has prediction bias."""
    prediction_counts = pd.Series(y_pred).value_counts()
    actual_counts = pd.Series(y_test).value_counts()
    
    logging.info("Prediction bias analysis:")
    logging.info(f"Actual distribution: {actual_counts.to_dict()}")
    logging.info(f"Predicted distribution: {prediction_counts.to_dict()}")
    
    # Calculate prediction ratio
    pred_spam_ratio = prediction_counts.get(1, 0) / len(y_pred)
    actual_spam_ratio = actual_counts.get(1, 0) / len(y_test)
    
    logging.info(f"Actual spam ratio: {actual_spam_ratio:.3f}")
    logging.info(f"Predicted spam ratio: {pred_spam_ratio:.3f}")
    
    if pred_spam_ratio < actual_spam_ratio * 0.5:
        logging.warning("Model appears to be biased towards predicting 'not spam'")

def save_results_to_file(results, output_path):
    """Save evaluation results to a text file."""
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("SMS Spam Filter Model Performance Evaluation\n")
        f.write("="*50 + "\n\n")

        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")
        if results['auc'] is not None:
            f.write(f"AUC: {results['auc']:.4f}\n")

        f.write("\nClassification Report:\n")
        f.write(results['report'])

        f.write("\nConfusion Matrix:\n")
        cm = results['confusion_matrix']
        f.write(f"True Negatives (TN): {cm[0][0]}\n")
        f.write(f"False Positives (FP): {cm[0][1]}\n")
        f.write(f"False Negatives (FN): {cm[1][0]}\n")
        f.write(f"True Positives (TP): {cm[1][1]}\n")

    logging.info(f"Evaluation results saved to {output_path}")

def main():
    """Main function to test model performance."""
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Paths
    model_path = "data/models/model.pkl"
    data_path = "data/processed/labeled_messages.csv"
    results_path = "results/evaluation_results.txt"
    cm_plot_path = "results/confusion_matrix.png"
    roc_plot_path = "results/roc_curve.png"

    # Load model and data
    model = load_model(model_path)
    X_test, y_test = load_data(data_path)

    # Evaluate model
    results = evaluate_model(model, X_test, y_test)

    # Plot and save results
    plot_confusion_matrix(results['confusion_matrix'], cm_plot_path)
    plot_roc_curve(results['fpr'], results['tpr'], results['auc'], roc_plot_path)

    # Analyze prediction errors
    error_analysis = analyze_errors(X_test, y_test, results['y_pred'])

    # Analyze model bias
    analyze_model_bias(y_test, results['y_pred'])

    # Save results to file
    save_results_to_file(results, results_path)

    logging.info("Model performance testing completed!")

if __name__ == "__main__":
    main()