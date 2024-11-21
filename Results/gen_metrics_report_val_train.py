import os
import pandas as pd
import numpy as np
import re
import json
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, accuracy_score
)

VALID_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema",
    "Foreign Body", "Lymphangiectasia", "Normal",
    "Polyp", "Ulcer", "Worms"
]

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def class_wise_metrics(y_true, y_pred):
    class_metrics = {}
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    for i, class_name in enumerate(VALID_CLASSES):
        true_binary = (y_true_classes == i).astype(int)
        pred_binary = (y_pred_classes == i).astype(int)

        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        specificity = specificity_score(true_binary, pred_binary)
        accuracy = accuracy_score(true_binary, pred_binary)
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'accuracy': accuracy,
            'auc': auc
        }
    return class_metrics

def generate_metrics_report(y_true, y_pred):
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    auc_roc_scores = []
    for i in range(len(VALID_CLASSES)):
        try:
            auc_roc_scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except ValueError:
            auc_roc_scores.append(0.0)
    mean_auc_roc = np.mean(auc_roc_scores)

    avg_precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    avg_sensitivity = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    avg_f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

    avg_specificity = np.mean([
        specificity_score(y_true_classes == i, y_pred_classes == i)
        for i in range(len(VALID_CLASSES))
    ])

    balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
    class_metrics = class_wise_metrics(y_true, y_pred)

    return {
        'mean_auc': mean_auc_roc,
        'balanced_accuracy': balanced_acc,
        'avg_precision': avg_precision,
        'avg_sensitivity': avg_sensitivity,
        'avg_f1': avg_f1,
        'avg_specificity': avg_specificity,
        'class_wise_metrics': class_metrics
    }

def save_metrics_report(report, output_path):
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)

def sanity_check(true_df, pred_df):
    def extract_filename(path):
        match = re.search(r'([^/\\]+\.jpg)$', path)
        return match.group(1) if match else None

    true_images = true_df['image_path']
    pred_images = pred_df['image_path']
    missing_images = set(true_images) - set(pred_images)
    extra_images = set(pred_images) - set(true_images)

    if missing_images or extra_images:
        return False, None

    pred_df = pred_df.set_index(pred_images)
    aligned_pred_df = pred_df.loc[true_images].reset_index(drop=True)
    required_columns_present = all(col in aligned_pred_df.columns for col in VALID_CLASSES)
    predicted_class_present = True
    no_missing_values = not aligned_pred_df.isnull().values.any()
    valid_classes = True
    no_duplicates = not aligned_pred_df['image_path'].duplicated().any()

    all_checks_passed = all([
        required_columns_present, predicted_class_present,
        no_missing_values, valid_classes, no_duplicates
    ])

    return all_checks_passed, aligned_pred_df

def process_files(true_filepath, pred_folder, output_folder):
    true_df = pd.read_excel(true_filepath)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for pred_file in os.listdir(pred_folder):
        try:
            pred_df = pd.read_excel(os.path.join(pred_folder, pred_file))
            format_correct, aligned_pred_df = sanity_check(true_df, pred_df)

            if format_correct:
                y_true = true_df[VALID_CLASSES].to_numpy()
                y_pred = aligned_pred_df[VALID_CLASSES].to_numpy()

                metrics_report = generate_metrics_report(y_true, y_pred)
            else:
                metrics_report = {
                    'mean_auc': None, 'balanced_accuracy': None,
                    'avg_precision': None, 'avg_sensitivity': None,
                    'avg_f1': None, 'avg_specificity': None,
                    'class_wise_metrics': {}
                }

            output_path = os.path.join(output_folder, f"{os.path.splitext(pred_file)[0]}_metrics.json")
            save_metrics_report(metrics_report, output_path)

        except Exception as e:
            print(f"Error processing file '{pred_file}': {e}")
            continue  

true_filepath = 'training_data.xlsx'
pred_folder = 'training_fixed'
output_folder = 'metrics_reports_train'

process_files(true_filepath, pred_folder, output_folder)
