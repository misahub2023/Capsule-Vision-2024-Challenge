import argparse
from tensorflow.keras.models import load_model
import numpy as np
from data_loader import get_data, load_test_data
from Eval_metrics_gen_excel import generate_classification_report, generate_auc_roc, save_predictions_to_excel, save_class_predictions_to_excel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification and Evaluation')
    parser.add_argument('--val_excel_path', type=str, default="D:/misahub/Dataset/Dataset/validation/validation_data.xlsx")
    parser.add_argument('--test_dir', type=str, default="D:/misahub/Test set for participants/Final test set/Images")
    parser.add_argument('--base_dir', type=str, default="D:/misahub/Dataset/Dataset")
    parser.add_argument('--model_path', type=str, default="model.keras")
    parser.add_argument('--output_val_predictions', type=str, default="validation_predictions.xlsx")
    parser.add_argument('--output_test_predictions', type=str, default="test_predictions.xlsx")
    parser.add_argument('--output_test_class_predictions', type=str, default="test_class_predictions.xlsx")
    parser.add_argument('--image_size', type=tuple, default=(32, 32))

    args = parser.parse_args()

    model = load_model(args.model_path)

    X_val, y_val, val_df = get_data(args.val_excel_path, base_dir=args.base_dir, image_size=args.image_size)
    y_val_pred = model.predict(X_val)

    classification_report_val = generate_classification_report(y_val, y_val_pred)
    auc_roc_scores_val = generate_auc_roc(y_val, y_val_pred)

    print("Validation Classification Report:")
    print(classification_report_val)

    print("Validation AUC-ROC Scores:")
    print(auc_roc_scores_val)

    save_predictions_to_excel(val_df['image_path'].values, y_val_pred, args.output_val_predictions)

    X_test, test_image_paths = load_test_data(args.test_dir, image_size=args.image_size)

    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)

    save_predictions_to_excel(test_image_paths, y_test_pred, args.output_test_predictions)
    save_class_predictions_to_excel(test_image_paths, y_test_pred_classes, args.output_test_class_predictions)
