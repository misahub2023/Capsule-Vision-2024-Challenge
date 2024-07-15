import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

def load_and_preprocess_image(base_dir, path, target_size):
    img = load_img(os.path.join(base_dir, path), target_size=target_size)
    img_array = img_to_array(img)
    return preprocess_input(img_array)

def data_generator(excel_path, base_dir, image_size=(32, 32), batch_size=4096):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    class_names = df.columns[2:]
    while True:
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_paths = df['image_path'].values[start:end]
            batch_labels = df[class_names].values[start:end]
            batch_images = [load_and_preprocess_image(base_dir, path, image_size) for path in batch_paths]
            yield np.array(batch_images), np.array(batch_labels)

def get_data_for_training(excel_path, base_dir, image_size=(32, 32)):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    class_names = df.columns[2:]
    X = np.array([load_and_preprocess_image(base_dir, path, image_size) for path in df['image_path'].values])
    y = df[class_names].values
    return X, y, df['image_path'].values

if __name__ == "__main__":
    train_excel_path = "/content/drive/MyDrive/Dataset/Dataset/training/training_data.xlsx"    
    val_excel_path = "/content/drive/MyDrive/Dataset/Dataset/validation/validation_data.xlsx"
    base_dir = "/content/drive/MyDrive/Dataset/Dataset"
    image_size = (32, 32)
    batch_size = 4096
    epochs = 10

    #Load training and validation data
    X_train, y_train, _ = get_data_for_training(train_excel_path, base_dir, image_size=image_size)
    print("train data loaded")
    X_val, y_val, image_paths = get_data_for_training(val_excel_path, base_dir, image_size=image_size)
    print("val data loaded")

    class_names = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weights_dict = dict(enumerate(class_weights))

    #Build the model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    #Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict
    )

    #Model Evalulation
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)

    results_df = pd.DataFrame({
        'Image_path': image_paths,
        'actual_class': y_val_classes,
        'predicted_class': y_pred_classes
    })
    results_df.to_excel("results.xlsx", index=False, sheet_name='Sheet1')

    results_file_path = os.path.join(base_dir, "results.xlsx")
    results_df.to_excel(results_file_path, index=False, sheet_name='Sheet1')

    #Confusion matrix and classification report
    print("Confusion Matrix:")
    cm = confusion_matrix(y_val_classes, y_pred_classes, normalize='true')
    print(cm)

    print("Classification Report:")
    print(classification_report(y_val_classes, y_pred_classes))

    #Normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    #AUC-ROC curve
    lb = LabelBinarizer()
    y_val_bin = lb.fit_transform(y_val_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred[:, i])
        roc_auc[i] = roc_auc_score(y_val_bin[:, i], y_pred[:, i])

    plt.figure()
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.legend(loc='lower right')
    plt.show()