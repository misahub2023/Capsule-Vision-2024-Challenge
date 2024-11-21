# Classification Benchmarking of the Capsule Vision 2024 Challenge
This repository contains scripts used in the performance evaluation of 6 artificial intelligence models for the classification of the [Capsule Vision 2024 Challenge Dataset](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469?file=48018562). Six classification-based pipelines have been trained, validated, and tested.
The models employed are:
* VGG19
* Xception
* ResNet50V2
* MobileNetV2
* InceptionV3
* InceptionResNetV2


## Dataset
Organize your dataset in the following structure:The dataset comprises gastroenterology images divided into 10 abnormality categories.
The dataset structure was as follows:
```
	Dataset/
	├── training/
	│   ├── Angioectasia/
	│   ├── Bleeding/
	│   ├── Erosion/
	│   ├── Erythema/
        │   ├── Foreign Body/
	│   ├── Lymphangiectasia/
	│   ├── Normal/
	│   ├── Polyp/
	│   ├── Ulcer/
	│   ├── Worms/
	│   ├── training_data.xlsx
	├── validation/
	│   ├── Angioectasia/
   	│   ├── Bleeding/
    	│   ├── Erosion/
	│   ├── Erythema/
   	│   ├── Foreign Body/
    	│   ├── Lymphangiectasia/
	│   ├── Normal/
   	│   ├── Polyp/
    	│   ├── Ulcer/
	│   ├── Worms/
   	│   ├── validation_data.xlsx/
    	├── Test set for us, with excel sheet and separated /
	   ├── Angioectasia/
	   ├── Bleeding/
	   ├── Erosion/
	   ├── Erythema/
           ├── Foreign Body/
	   ├── Lymphangiectasia/
	   ├── Normal/
	   ├── Polyp/
	   ├── Ulcer/
	   ├── Worms/
	   ├── test_data.xlsx
	
```	
## Benchmarking Code Usage:

1. Function defined to load images and labels from excel file data, and converting images into numpy array
	```
	def load_and_preprocess_image(full_path):
    		img = load_img(full_path)
    		img_array = img_to_array(img)
	    	return img_array
	
	def get_data(excel_path, base_dir):
    		df = pd.read_excel(excel_path)
    		df = df.dropna(subset=['image_path'])
    		class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    		X = np.array([load_and_preprocess_image(os.path.join(base_dir,path.replace("\\","/"))) for path in df['image_path'].values])
    		y = df[class_columns].values
    		return X, y, df
	```
  Loading testing images and processing them.
	```
	def get_test_data(excel_path,base_dir):
    		df = pd.read_excel(excel_path)
    		df = df.dropna(subset=['image_path'])
	        class_column=['class_label']
     		x = np.array([load_and_preprocess_image(os.path.join(base_dir,path.replace("\\","/"))) for path in df['image_path'].values])
                y= df[class_column].values
      		return x,y,df
	```
2. Define paths to excel files and base directory.

	```
	train_excel_path = "/workspace/Documents/Pallavi/Dataset/training/training_data.xlsx"
	val_excel_path = "/workspace/Documents/Pallavi/Dataset/validation/validation_data.xlsx"
	test_excel_path = "/workspace/Documents/Pallavi/Dataset/Test set for us, with excel sheet and seperated/test_data.xlsx" 
	base_dir = "/workspace/Documents/Pallavi/Dataset"

	```
3. Loading training, validation and testing images.

	```
	X_train, y_train, train_df = get_data(train_excel_path, base_dir)
	X_val, y_val, val_df = get_data(val_excel_path, base_dir)
	X_test, y_test, test_df = get_test_data(test_excel_path,base_dir)
	```
4. Defining categories for classification

	```
	categories = ['Angioectasia','Bleeding', 'Erosion','Erythema','Foreign Body',
                            'Lymphangiectasia','Normal','Polyp','Ulcer','Worms']
	```
5. Augmentation - For benchmarking, model is trained without augmentation for that following are set.

	```
	datagen=ImageDataGenerator(
    		featurewise_center=False,
    		samplewise_center=False,
    		featurewise_std_normalization=False,
    		samplewise_std_normalization=False,
    		zca_whitening=False,
    		zca_epsilon=1e-06, # it is default value. If set to none will disable whitening process
    		# it applies only when zca_whitening is True
    		rotation_range=0,
    		width_shift_range=0.0,
    		height_shift_range=0.0,
    		brightness_range=None,
    		shear_range=0.0,
    		zoom_range=0.0,
    		channel_shift_range=0.0,
    		fill_mode='nearest',# On performing transformations some pixels may fall outside the boundaries of the input image. So, fill_mode controls how these newly created pixels are filled in by default it is nearest. if set to 0.0 or none, it will cause error
		cval=0.0,
		horizontal_flip=False,
     		vertical_flip=False,
    		# rescale=1./255, # normalizes the pixel values. By default it is none.
    		# If set to 0, it make the image useless for model training, as all info will be lost
    		preprocessing_function=None,
    		data_format=None,
    		validation_split=0.0,
    		# interpolation_order=1,
    		dtype=None)

	datagen.fit(X_train,augment=False,rounds=1,seed=42)


	```
	
6. Baseline model is initialized which is trained on the *ImageNet* dataset, its top layers of the network should not be included, and defines the input images shape.
In same way other models (e.g., `VGG19`, `Xception`, `ResNet50V2`, `MobileNetV2`, `InceptionV3`, `InceptionResNetV2`) can be intialized.
	```
	base_model= MobileNetV2(weights='imagenet', include_top=False,input_shape=(224,224,3))
	```

7. After initializing base model, a custom classification head is added on top of the model so that it works for our dataset.

	```
	# Get the output from the base model
	x = base_model.output
	# Flatten the output to prepare it for the Dense layer
	x = Flatten()(x)
	# Define the final prediction layer with 10 units (classes), using softmax activation
	predictions = Dense(10, activation='softmax',
	use_bias=True, # Include a bias term in the layer
    	kernel_initializer="glorot_uniform", # it is by default. Also known as Xavier uniform
        # suitable for layers using tanh or softmax activation functions
    	bias_initializer="zeros", # Initialize biases to zero
    	kernel_regularizer=None,
    	bias_regularizer=None,
    	activity_regularizer=None,
    	kernel_constraint=None,
    	bias_constraint=None,)(x)


	```

8. Combining base model input and the custom prediction layers 

	```
	model = Model(inputs=base_model.input, outputs=predictions)

	```
9. Freezing all layers in the base model to prevent them from being updating during training
	```
	for layer in base_model.layers:
    	    layer.trainable = False
	```
10. Compiling the Model. Parameters are set for this are -

* `--optimizer` - Name of the optimizer to use (e.g., `Adam`,`SGD`,`RMSprop`).
* `--loss` - Loss function to use for multiclassification is `categorical_crossentropy`.
* `--metrics` - Metrics functions are defined to evaluate the performance of a model during training, validation, and testing. These are for monitoring purposes. 
* `--weighted_metrics` - Weighted metrics are extension of regular metrics. This is useful when dealing with *class imbalance* or assigning more importance to certain data points.

	```
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy','AUC','Precision','Recall'],
    		weighted_metrics=['accuracy','Precision','Recall','AUC'],
    		loss_weights=None,
    		# weighted_metrics=None,
    		run_eagerly=None,
    		steps_per_execution=None,
    		jit_compile=None,
    		# pss_evaluation_shards=0
              )
	```
11. Total number of trainable and non-trainable parameters are calculated in the model.

	```
	trainable_params = sum(K.count_params(layer) for layer in model.trainable_weights)
	non_trainable_params = sum(K.count_params(layer) for layer in model.non_trainable_weights)
	print("Trainable params: ", trainable_params)
	print("Non-trainable params: ", non_trainable_params)
	print("Total params:",non_trainable_params+trainable_params)

	```

12. Creating CSV logger file to store all logs of model while training.

	```
	csv_logger = CSVLogger("model_history_log.csv", append=True)
	```

13. Training the model.

* `--batch_size` - batch size for training (default is 32).
* `--epochs` - Number of epochs to train.
* `--callbacks` - File to store logs of model training.

	```
	model.fit(datagen.flow(X_train, y_train, batch_size=256,shuffle=True,
                       seed=42,
                       save_to_dir=None,save_prefix='',
                       save_format='png',
                       subset=None),
          epochs=250, validation_data=(X_val, y_val),
          verbose=1,callbacks=csv_logger,shuffle=True,class_weight=None,
          sample_weight=None,
          steps_per_epoch=None,
          validation_batch_size=64,
          )

	```

14. Evaluation of the trained model on each classification and taking average of their metrics results to conclude overall result of the trained model. Metrics evaluated for the model are `mean_auc`,`mean_specificity`,`mean_average_precision`,`mean_sensitivity`,`mean_f1_score`,`balanced_accuracy`.

	```
	# Evaluation Metrics Generate Excel
	def save_predictions_to_excel(image_paths, y_pred, output_path,Dataset=None):
        	class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
        	y_pred_classes = np.argmax(y_pred, axis=1)
        	predicted_class_names = [class_columns[i] for i in y_pred_classes]
        	df_prob = pd.DataFrame(y_pred, columns=class_columns)
        	df_prob.insert(0, 'image_path', image_paths)
        	# Add Dataset column if provided
        	if Dataset is not None:
            	df_prob.insert(1, 'Dataset', Dataset)
        	df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
        	df_merged = pd.merge(df_prob, df_class, on='image_path')
        	df_merged.to_excel(output_path, index=False)
	def calculate_specificity(y_true, y_pred):
        		tn = np.sum((y_true == 0) & (y_pred == 0))
        	fp = np.sum((y_true == 0) & (y_pred == 1))
        	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        	return specificity
	def generate_metrics_report(y_true, y_pred):
        	class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
        	metrics_report = {}

        	y_true_classes = np.argmax(y_true, axis=1)
		# y_true_classes = y_true
        	y_pred_classes = np.argmax(y_pred, axis=1)
		class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_columns, output_dict=True, zero_division=0)

        	auc_roc_scores = {}
        	for i, class_name in enumerate(class_columns):
            		auc_roc_scores[class_name] = roc_auc_score(y_true[:, i], y_pred[:, i])

        	mean_auc_roc = np.mean(list(auc_roc_scores.values()))
        	auc_roc_scores['mean_auc'] = mean_auc_roc

        	specificity_scores = {}
        	for i, class_name in enumerate(class_columns):
            	specificity_scores[class_name] = calculate_specificity(y_true[:, i], (y_pred[:, i] > 0.5).astype(int))

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
            	sensitivity_scores[class_name] = recall_score(y_true[:, i], (y_pred[:, i] > 0.5).astype(int), average='binary', zero_division=0)

        	mean_sensitivity = np.mean(list(sensitivity_scores.values()))
        	sensitivity_scores['mean_sensitivity'] = mean_sensitivity

        	f1_scores = {}
        	for i, class_name in enumerate(class_columns):
            	f1_scores[class_name] = f1_score(y_true[:, i], (y_pred[:, i] > 0.5).astype(int), average='binary', zero_division=0)

        	mean_f1_score = np.mean(list(f1_scores.values()))
        	f1_scores['mean_f1_score'] = mean_f1_score
        	balanced_accuracy_scores = balanced_accuracy_score(y_true_classes, y_pred_classes)

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
        	metrics_report['balanced_accuracy'] = balanced_accuracy_scores
        
        	metrics_report_json = json.dumps(metrics_report, indent=4)
        	return metrics_report_json

	```

15. Prediction on testing data and generating metrics report. This is done for training and validation data also.

	```
	y_test_pred = model.predict(X_test)
	test_metrics_rep=generate_metrics_report(test_labels,y_test_pred)
	print(test_metrics_rep)

	```
16. Saving the testing data predictions to excel.

	```
	output_test_predictions="test_excel.xlsx"
	save_predictions_to_excel(test_df['image_path'],y_test_pred,output_test_predictions)

	``` 

## Setup Used for Evaluation
All the models were trained for a total of 250 epochs, without any preprocessing or modification. The codes were run using 40GB DGX A100 NVIDIA GPU workstation available at the Department of Electronics and Communication Engineering, Indira Gandhi Technical University for Women, New Delhi, India.


## Results
The results and findings will be released in the form of a research paper soon, the preprint has been released and can be accessed at [link](https://arxiv.org/abs/2408.04940).


## Contributions
Pallavi Sharma contributed in developing the benchmarking pipeline and developing the github repository for it.