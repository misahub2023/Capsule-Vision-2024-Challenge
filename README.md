![ChallengeHeader](https://github.com/user-attachments/assets/e75f510b-02a8-4fec-b133-11f4ab6c828d)
# Capsule Vision Challenge 2024: Multi-Class Abnormality Classification for Video Capsule Endoscopy
- [Registration form](https://forms.gle/QemRirWysnGoGrKM6) (closed now)
- [Challenge Hosting Website](https://misahub.in/cv2024.html)
- [Challenge ArXiv](https://arxiv.org/abs/2408.04940)
- [Challenge github repository](https://github.com/misahub2023/Capsule-Vision-2024-Challenge)
- [Training and Validation Dataset Link](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469?file=48018562)
- [Testing Dataset Link](https://figshare.com/articles/dataset/Testing_Dataset_of_Capsule_Vision_2024_Challenge/27200664?file=49717386)
- [Sample Report Overleaf](https://www.overleaf.com/read/kwhvpznnbzwb#26d62a)
- [Submission Sanity Checker](https://capsulevisionchallengesanitychecker.streamlit.app)
## Table of Content
- [Challenge Overview](#challenge-overview)
- [Challenge Timeline](#challenge-timeline)
- [Dataset](#dataset)
  - [Dataset Structure](#dataset-structure)
- [Sample Scripts for Participants](#sample-scripts-for-participants)
  - [Data_loader Script](#data_loader)
  - [Eval_metrics_gen_excel](#eval_metrics_gen_excel)
  - [Evaluate_model](#evaluate_model)
- [Sample Evaluation by Organizing members](#sample-evaluation-by-organizing-members)
- [Submission Format](#submission-format)
- [Results](https://github.com/misahub2023/Capsule-Vision-2024-Challenge/edit/main/README.md#results)
- [Benchmarking Experiments](https://github.com/misahub2023/Capsule-Vision-2024-Challenge/tree/main/Benchmarking_Experiments)
## Challenge Overview
The aim of the challenge was to provide an opportunity
for the development, testing and evaluation of AI models
for automatic classification of abnormalities captured in
VCE video frames. It promoted the development of vendor-independent and
generalized AI-based models for automatic abnormality
classification pipeline with 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body,
lymphangiectasia, polyp, ulcer, worms, and normal.
## Challenge Timeline
- **Launch of challenge:** August 15, 2024
- **Registration:** August 15, 2024 - October 10, 2024
- **Release of Training Data:** August 15, 2024
- **Release of Test Data and Result submission:** October 11, 2024 - October 25, 2024
- **Result analysis by the organizing team:** October 26, 2024 - November 24, 2024
- **Announcement of results for all teams:** November 25, 2024
## Dataset 
The training and validation dataset were developed using
three publicly available (SEE-AI project dataset, KID,
and Kvasir-Capsule dataset) and one private dataset (AIIMS) VCE datasets. The training and validation dataset
consisted of 37,607 and 16,132 VCE frames respectively mapped to 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body, lymphangiectasia, polyp, ulcer, worms,
and normal.
| Type of Data | Source Dataset | Angioectasia | Bleeding | Erosion | Erythema | Foreign Body | Lymphangiectasia | Normal | Polyp | Ulcer | Worms |
|--------------|----------------|--------------|----------|---------|----------|---------------|------------------|--------|-------|-------|-------|
| Training     | KID            | 18           | 3        | 0       | 0        | 0             | 6                | 315    | 34    | 0     | 0     |
|              | KVASIR         | 606          | 312      | 354     | 111      | 543           | 414              | 24036  | 38    | 597   | 0     |
|              | SEE-AI         | 530          | 519      | 2340    | 580      | 249           | 376              | 4312   | 1090  | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 66    | 158   |
| **Total Frames** |                | **1154**     | **834**  | **2694**| **691**  | **792**       | **796**          | **28663**| **1162**| **663**| **158** |
| Validation   | KID            | 9            | 2        | 0       | 0        | 0             | 3                | 136    | 15    | 0     | 0     |
|              | KVASIR         | 260          | 134      | 152     | 48       | 233           | 178              | 10302  | 17    | 257   | 0     |
|              | SEE-AI         | 228          | 223      | 1003    | 249      | 107           | 162              | 1849   | 468   | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 29    | 68    |
| **Total Frames** |                | **497**      | **359**  | **1155**| **297**  | **340**       | **343**          | **12287**| **500** | **286** | **68** |

### Dataset Structure
The images are organized into their respective classes for both the training and validation datasets as shown below:
```bash
Dataset/
├── training
│   ├── Angioectasia
│   ├── Bleeding
│   ├── Erosion
│   ├── Erythema
│   ├── Foreign Body
│   ├── Lymphangiectasia
│   ├── Normal
│   ├── Polyp
│   ├── Ulcer
│   └── Worms
│   └── training_data.xlsx
└── validation
    ├── Angioectasia
    ├── Bleeding
    ├── Erosion
    ├── Erythema
    ├── Foreign Body
    ├── Lymphangiectasia
    ├── Normal
    ├── Polyp
    ├── Ulcer
    └── Worms
    └── validation_data.xlsx
```
## Sample Scripts for Participants
### Data_loader
The [Data_loader.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Data_loader.py) script fetches the data from Figshare, unzips and saves it in the current directory.

```bash
python sample_codes_for_participants/Data_loader.py
```
### Eval_metrics_gen_excel 
The [Eval_metrics_gen_excel.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Eval_metrics_gen_excel.py) script contains 2 functions:

#### save_predictions_to_excel
  
The `save_predictions_to_excel` function processes the predicted probabilities for a set of images, determines the most likely class for each image, and then saves the results (including both the predicted probabilities and the predicted classes) to an Excel file.
The function takes 3 parameters:
   - `image_paths`: A list of paths to the images. Each path corresponds to an image that was used for prediction.
   - `y_pred`: A numpy array containing the predicted probabilities for each class. Each row corresponds to an image, and each column corresponds to a class.
   - `output_path`: The file path where the Excel file will be saved.

A sample of the excel file which will be generated using this function is available [here](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample%20evaluation%20by%20organizing%20members/VGG16/validation_excel.xlsx)

The generated excel file for the train, validation, and test data was to be submitted through for evaluation. [Check here](#submission-format)
Note: The y_pred array should have the predicted probabilites in the order: `['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']`

#### generate_metrics_report
  
The `generate_metrics_report` function generates all the relevant metrics for evaluating a multi-class classification, including classwise and aggregate specificity, ROC AUC scores, precision-recall scores, sensitivity, F1 score, mean AUC and balanced accuracy score. This function can be used to evaluate the performance of a trained model on validation data.

The function takes 2 parameters:
  - y_true: The ground truth multi-class labels in one-hot encoded format.
  - y_pred: The predicted probabilities for each class.

Returns: A JSON string containing the detailed metrics report.

Note: The y_pred and y_true array should have the predicted probabilites in the order: `['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']`

#### Usage
To run this script, you'll need to have the following libraries installed:

1. **NumPy**: For numerical operations and array handling.
   - Install using: `pip install numpy`

2. **Pandas**: For data manipulation and analysis.
   - Install using: `pip install pandas`

3. **Scikit-learn**: For machine learning metrics and utilities.
   - Provides functions such as `classification_report`, `roc_auc_score`, `precision_recall_curve`, `recall_score`, and `f1_score`.
   - Install using: `pip install scikit-learn`

4. **JSON**: For parsing JSON data (included in Python standard library, no installation required).

To install all required libraries, you can use the following command:

```bash
pip install numpy pandas scikit-learn
```
### Evaluate_model
The [Evaluate_model.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Evaluate_model.py) script is a sample script which shows the usage of the functions from the [Eval_metrics_gen_excel.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Eval_metrics_gen_excel.py) script. A VGG16 model has been evaluated in this script, participants can take inspiration from this for their own submissions.

The sample script that shows the metrics and creates the prediction excel file can be called via
```bash
python sample_codes_for_participants/Evaluate_model.py
```

## Sample Evaluation by organizing members

This directory contains extensive analysis of the dataset along with the evaluation of Custom CNN, Support vector machines, ResNet50 and VGG16 on the training and the validation data.

## Submission Format

Each team is required to submit their results in an EMAIL with the following structure to ask.misahub@gmail.com
- The email should contain:
  - Challenge name and Team name as the **SUBJECT
LINE**.
  - Team member names and affiliation in the **BODY
OF THE EMAIL**.
  - Contact number and email address in the **BODY
OF THE EMAIL**.
    A link of the github repository in public mode in
the **BODY OF THE EMAIL**.
  - A link of their report on any open preprint server of
their choice (ArXiv, Authorea, BioRxiv, Figshare
etc) in the **BODY OF THE EMAIL**. The report should be in the latex format given [here](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/Sample_report_format/Sample%20report%20for%20submission.zip).
  - Generated Excel sheet of the predicted train, validation, and test dataset (in xlsx format) as an attachment.
    The format of the excel sheet should match the sample given [here](https://capsulevisionchallengesanitychecker.streamlit.app). Please use the sanity checker to validate your file before submission.
    Any other format was **NOT** accepted and led to disqualification. 
- The github repository in public mode should contain the
following:
  - Developed code for training, validation, and testing
in .py / .mat etc in readable format with comments.
  - Stored model, associated weights or files (optional).
  - Any utils or assets or config. or checkpoints.

The submitted excel files were evaluated using a semi-automated script to generate the relevant metrics against the true labels on the basis of which the participants were ranked.

Following checklist was used to select the top three
winning teams:
- Combined metric (average of mean AUC and balanced accuracy) on
testing dataset.
## Results 
The script gen_metrics_report_val_train.py was used to generate the metrics for training and validation dataset whereas the script gen_metrics_test.py was used for the test set metrics. The submitted_excel_files folder contains the excel files received from the participants for each set, it contains the predicted probability for each class. The generated metrics for each team are present in metrics_reports folder which is further divided into train, val and test. training_data.xlsx, validation_data.xlsx and test_data.xlsx are the true files used in the scripts. 

### Usage
#### Requirements

Before running the scripts, install the required Python dependencies. Use the provided `requirements.txt` file:

#### Installation

```bash
pip install -r requirements.txt
```

---

#### **Training/Validation Metrics Evaluation**

This script processes training data and generates metrics reports for model predictions. It aligns prediction files with the ground truth data, computes metrics, and saves the results in a JSON format.

#### Usage

```bash
python training_metrics.py <true_filepath> <pred_folder> <output_folder>
```

#### Parameters
- **`<true_filepath>`**: Path to the Excel file containing ground truth data (e.g., `training_data.xlsx`).
- **`<pred_folder>`**: Path to the folder containing prediction Excel files.
- **`<output_folder>`**: Path to the folder where metrics reports will be saved.

#### Example

```bash
python training_metrics.py training_data.xlsx training_fixed metrics_reports_train
```

---

#### **Testing Metrics Evaluation**

This script performs similar operations for testing data. It verifies the format of prediction files, computes metrics, and saves the reports.

#### Usage

```bash
python testing_metrics.py <true_filepath> <pred_folder> <output_folder>
```

#### Parameters
- **`<true_filepath>`**: Path to the Excel file containing ground truth data (e.g., `test_data.xlsx`).
- **`<pred_folder>`**: Path to the folder containing prediction Excel files.
- **`<output_folder>`**: Path to the folder where metrics reports will be saved.

#### Example

```bash
python testing_metrics.py test_data.xlsx testing metrics_reports_test
```

---

#### Output

For both scripts, the output is a set of JSON files saved in the specified output folder. Each file corresponds to a prediction file and contains:
- **Overall Metrics**:
  - Mean AUC
  - Balanced Accuracy
  - Average Precision
  - Average Sensitivity
  - Average F1 Score
  - Average Specificity
- **Class-Wise Metrics**:
  - Precision, Recall, F1 Score, Specificity, Accuracy, and AUC for each class.


## Citation
Please use the following citations for citing our work.

- Challenge ArXiv
  
@misc{handa2024capsulevision2024challenge,
      title={Capsule Vision 2024 Challenge: Multi-Class Abnormality Classification for Video Capsule Endoscopy}, 
      author={Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Manas Dhir and Deepti Chhabra and Shreshtha Jha and Pallavi Sharma and Vijay Thakur and Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Raman},
      year={2024},
      eprint={2408.04940},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04940}, 
}

@article{Handa2024training,
author = "Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Deepti Chhabra and Shreshtha Jha and Pallavi Sharma and Vijav Thakur and Manas Dhir and Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Raman",
title = "{Training and Validation Dataset of Capsule Vision 2024 Challenge}",
year = "2024",
month = "7",
url = "https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469",
doi = "10.6084/m9.figshare.26403469.v2"
} 

@article{Handa2024testing,
author = "Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Deepti Chhabra and Shreshtha Jha and Manas Dhir and Pallavi Sharma and Vijav Thakur and Dr. Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Ramanathan",
title = "{Testing Dataset of Capsule Vision 2024 Challenge}",
year = "2024",
month = "10",
url = "https://figshare.com/articles/dataset/Testing_Dataset_of_Capsule_Vision_2024_Challenge/27200664",
doi = "10.6084/m9.figshare.27200664.v3"
}


















  
