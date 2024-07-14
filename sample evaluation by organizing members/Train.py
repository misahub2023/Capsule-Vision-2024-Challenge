import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import tensorflow as tf

from data_loader import get_data

def main(args):
    X_train, y_train, _ = get_data(args.train_excel_path, args.base_dir, image_size=args.image_size)
    X_val, y_val, _ = get_data(args.val_excel_path, args.base_dir, image_size=args.image_size)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weights_dict = dict(enumerate(class_weights))
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(y_train.shape[1], activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weights_dict)

    model.save('model.keras')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VGG16 model for image classification')
    parser.add_argument('--train_excel_path', type=str, default="D:/misahub/Dataset/Dataset/training/training_data.xlsx")
    parser.add_argument('--val_excel_path', type=str, default="D:/misahub/Dataset/Dataset/validation/validation_data.xlsx")
    parser.add_argument('--base_dir', type=str, default="D:/misahub/Dataset/Dataset")
    parser.add_argument('--image_size', type=tuple, default=(32, 32))
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    main(args)
