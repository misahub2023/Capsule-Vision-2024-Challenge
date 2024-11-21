import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    roc_auc_score, classification_report, precision_recall_curve, 
    recall_score, f1_score, auc, balanced_accuracy_score
)

VALID_CLASSES = [
    'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 
    'Foreign Body', 'Lymphangiectasia', 'Normal', 
    'Polyp', 'Ulcer', 'Worms'
]

def save_predictions_to_excel(image_paths, y_pred, output_path):
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_class_names = [VALID_CLASSES[i] for i in y_pred_classes]
    df_prob = pd.DataFrame(y_pred, columns=VALID_CLASSES)
    df_prob.insert(0, 'image_path', image_paths)
    df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df_merged = pd.merge(df_prob, df_class, on='image_path')
    df_merged.to_excel(output_path, index=False)

def calculate_specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def generate_metrics_report(y_true, y_pred):
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    class_report = classification_report(
        y_true_classes, y_pred_classes, target_names=VALID_CLASSES, 
        output_dict=True, zero_division=0
    )
    auc_roc_scores = {}
    for i, class_name in enumerate(VALID_CLASSES):
        try:
            auc_roc_scores[class_name] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc_roc_scores[class_name] = 0.0
    mean_auc_roc = np.mean(list(auc_roc_scores.values()))
    specificity_scores = {}
    for i, class_name in enumerate(VALID_CLASSES):
        specificity_scores[class_name] = calculate_specificity(y_true[:, i], y_pred_classes == i)
    mean_specificity = np.mean(list(specificity_scores.values()))
    average_precision_scores = {}
    for i, class_name in enumerate(VALID_CLASSES):
        try:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            average_precision_scores[class_name] = auc(recall, precision)
        except ValueError:
            average_precision_scores[class_name] = 0.0
    mean_average_precision = np.mean(list(average_precision_scores.values()))
    sensitivity_scores = {class_name: class_report[class_name]['recall'] for class_name in VALID_CLASSES}
    mean_sensitivity = np.mean(list(sensitivity_scores.values()))
    f1_scores = {class_name: class_report[class_name]['f1-score'] for class_name in VALID_CLASSES}
    mean_f1_score = np.mean(list(f1_scores.values()))
    balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
    metrics_report = {
        'class_report': class_report,
        'auc_roc_scores': auc_roc_scores,
        'specificity_scores': specificity_scores,
        'average_precision_scores': average_precision_scores,
        'sensitivity_scores': sensitivity_scores,
        'f1_scores': f1_scores,
        'mean_auc': mean_auc_roc,
        'mean_specificity': mean_specificity,
        'mean_average_precision': mean_average_precision,
        'mean_sensitivity': mean_sensitivity,
        'mean_f1_score': mean_f1_score,
        'balanced_accuracy': balanced_acc
    }
    return json.dumps(metrics_report, indent=4)

def process_predictions(image_paths, y_true, y_pred, output_excel_path):
    save_predictions_to_excel(image_paths, y_pred, output_excel_path)
    report = generate_metrics_report(y_true, y_pred)
    print(report)
