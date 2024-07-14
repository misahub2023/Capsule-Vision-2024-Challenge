import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_image(path, target_size):
    img = load_img(path, target_size=target_size)
    img_array = img_to_array(img)
    preprocessed_img = preprocess_input(img_array)
    flattened_img = preprocessed_img.flatten()
    return flattened_img

def get_data_for_training(excel_path, image_size=(32, 32)):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])

    df = df.drop(columns=['Dataset'])

    X = []
    y = []
    image_paths = []

    for idx, row in df.iterrows():
        image_path = row['image_path']
        label = row.drop('image_path').idxmax()  

        image = load_and_preprocess_image(image_path, image_size)

        X.append(image)
        y.append(label)
        image_paths.append(image_path)

    X = np.array(X)
    y = np.array(y)

    return X, y, image_paths

if __name__ == "__main__":
    train_excel_path = r"E:\ML SELF CODES\data check\Dataset\training\training_data.xlsx"
    val_excel_path = r"E:\ML SELF CODES\data check\Dataset\validation\validation_data.xlsx"
    image_size = (32, 32)  
    batch_size = 4096

    X_train, y_train, _ = get_data_for_training(train_excel_path, image_size=image_size)
    print("train data loaded")
    X_val, y_val, image_paths = get_data_for_training(val_excel_path, image_size=image_size)
    print("val data loaded")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print("scaling complete")

    # Initialize and train the SVM model
    svm_model = SVC(kernel='rbf', probability=True)  
    svm_model.fit(X_train, y_train)
    print("training complete")

    #printing accuracy
    y_pred = svm_model.predict(X_val)
    y_pred_proba = svm_model.predict_proba(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy}")

    # Save validation predictions to an Excel file
    results_df = pd.DataFrame({
        'Image_path': image_paths,
        'actual_class': y_val,
        'predicted_class': y_pred
    })
    results_df.to_excel("results.xlsx", index=False, sheet_name='Sheet1')

    # Print the confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred, normalize='true')
    print(cm)

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # printing the normalized confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # Plot AUC-ROC curve
    lb = LabelBinarizer()
    y_val_bin = lb.fit_transform(y_val)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = roc_auc_score(y_val_bin[:, i], y_pred_proba[:, i])

    plt.figure()
    for i in range(len(lb.classes_)):
        plt.plot(fpr[i], tpr[i], label=f'Class {lb.classes_[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
