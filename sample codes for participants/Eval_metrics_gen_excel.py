import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def generate_classification_report(y_true, y_pred):
    class_columns=['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_columns, output_dict=True)
    return report

def generate_auc_roc(y_true, y_pred):
    class_columns=['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    auc_roc_scores = {}
    for i, class_name in enumerate(class_columns):
        auc_roc_scores[class_name] = roc_auc_score(y_true[:, i], y_pred[:, i])
    mean_auc = np.mean(list(auc_roc_scores.values()))
    auc_roc_scores['mean_auc'] = mean_auc
    return auc_roc_scores

def save_predictions_to_excel(image_paths,y_pred,output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    df = pd.DataFrame(y_pred, columns=class_columns)
    df.insert(0, 'image_path', image_paths)
    df.to_excel(output_path, index=False)

def save_class_predictions_to_excel(image_paths,y_pred_classes,output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    df = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df.to_excel(output_path, index=False)