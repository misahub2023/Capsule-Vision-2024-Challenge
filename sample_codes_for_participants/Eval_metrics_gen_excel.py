import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import json
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score


def save_predictions_to_excel(image_paths, y_pred, output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    df_prob = pd.DataFrame(y_pred, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df_merged = pd.merge(df_prob, df_class, on='image_path')
    df_merged.to_excel(output_path, index=False)


def calculate_specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def generate_metrics_report(y_true, y_pred):
    class_columns=['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    metrics_report = {}
    
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_columns, output_dict=True)
    
    auc_roc_scores = {}
    for i, class_name in enumerate(class_columns):
        auc_roc_scores[class_name] = roc_auc_score(y_true[:, i], y_pred[:, i])
    
    mean_auc_roc = np.mean(list(auc_roc_scores.values()))
    auc_roc_scores['mean_auc'] = mean_auc_roc
    
    specificity_scores = {}
    for i, class_name in enumerate(class_columns):
        specificity_scores[class_name] = calculate_specificity(y_true[:, i], np.argmax(y_pred, axis=1))
    
    mean_specificity = np.mean(list(specificity_scores.values()))
    specificity_scores['mean_specificity'] = mean_specificity
    
    average_precision_scores = {}
    for i, class_name in enumerate(class_columns):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision_scores[class_name] = auc(recall, precision)
    
    mean_average_precision = np.mean(list(average_precision_scores.values()))
    average_precision_scores['mean_average_precision'] = mean_average_precision
    
    sensitivity_scores = {}
    for i, class_name in enumerate(class_columns):
        sensitivity_scores[class_name] = recall_score(y_true[:, i], np.argmax(y_pred, axis=1), average='macro')
    
    mean_sensitivity = np.mean(list(sensitivity_scores.values()))
    sensitivity_scores['mean_sensitivity'] = mean_sensitivity
    
    f1_scores = {}
    for i, class_name in enumerate(class_columns):
        f1_scores[class_name] = f1_score(y_true[:, i], np.argmax(y_pred, axis=1), average='macro')
    
    mean_f1_score = np.mean(list(f1_scores.values()))
    f1_scores['mean_f1_score'] = mean_f1_score
    
    metrics_report.update(class_report)
    metrics_report['auc_roc_scores'] = auc_roc_scores
    metrics_report['specificity_scores'] = specificity_scores
    metrics_report['average_precision_scores'] = average_precision_scores
    metrics_report['sensitivity_scores'] = sensitivity_scores
    metrics_report['f1_scores'] = f1_scores
    metrics_report['mean_auc'] = mean_auc_roc
    metrics_report['mean_specificity'] = mean_specificity
    metrics_report['mean_average_precision'] = mean_average_precision
    metrics_report['mean_sensitivity'] = mean_sensitivity
    metrics_report['mean_f1_score'] = mean_f1_score
    
    metrics_report_json = json.dumps(metrics_report, indent=4)
    return metrics_report_json



