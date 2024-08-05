![ChallengeHeader](https://github.com/user-attachments/assets/e75f510b-02a8-4fec-b133-11f4ab6c828d)
# Capsule Vision Challenge 2024: Multi-Class Abnormality Classification for Video Capsule Endoscopy
- [Challenge Website](https://misahub.in/cv2024.html)
- [Sample Report Overleaf](https://www.overleaf.com/project/668edec29a1be231946e844e)
- [Training and Validation Dataset Link](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469?file=48018562)
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
- [Contributions](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/edit/main/README.md#contributions)
## Challenge Overview
The aim of the challenge is to provide an opportunity
for the development, testing and evaluation of AI models
for automatic classification of abnormalities captured in
VCE video frames. It promotes the development of vendor-independent and
generalized AI-based models for automatic abnormality
classification pipeline with 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body,
lymphangiectasia, polyp, ulcer, worms, and normal.
## Challenge Timeline
- **Launch of challenge:** August 15, 2024
- **Registration open and closes:** August 15, 2024 - October 10, 2024
- **Release of Training Data:** August 15, 2024
- **Release of Test Data and Result submission open and closes:** October 11, 2024 - October 25, 2024
- **Result analysis by the organizing team:** October 26, 2024 - November 24, 2024
- **Announcement of results for all teams:** November 25, 2024
## Dataset 
The training and validation dataset has been developed using
three publicly available (SEE-AI project dataset, KID,
and Kvasir-Capsule dataset) and one private dataset (AIIMS) VCE datasets. The training and validation dataset
consist of 37,607 and 16,132 VCE frames respectively mapped to 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body, lymphangiectasia, polyp, ulcer, worms,
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
!python Data_loader.py
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

The generated excel file for the test data is to be submitted through for evaluation.[Check here](#submission-format)
Note: The y_pred array should have the predicted probabilites in the order: `['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']`

#### generate_metrics_report
  
The `generate_metrics_report` function generates all the relevant metrics for evaluating a multi-class classification, including classwise and aggregate specificity, ROC AUC scores, precision-recall scores, sensitivity, and F1 scores. This function can be used to evaluate the performance of a trained model on validation data.

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

## Sample Evaluation by organizing members

This directory contains extensive analysis of the dataset along with the evaluation of Custom CNN, Support vector machines, ResNet50 and VGG16 on the training and the validation data.

## Submission Format

Each team is required to submit their results via EMAIL to
ask.misahub@gmail.com with following rules in mind.
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
etc) in the **BODY OF THE EMAIL**. The report should be in the latex format given [here](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/Sample%20report%20for%20submission.zip).
  - Generated excel sheet (in xlsx format) as an attachment.
    The excel sheet should be in the same format as [this](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample%20evaluation%20by%20organizing%20members/VGG16/validation_excel.xlsx) for the test data and can be generated from the [Eval_metrics_gen_excel.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Eval_metrics_gen_excel.py) script. Any other format will **NOT** be accepted and will lead to disqualification.
- The github repository in public mode should contain the
following:
  - Developed code for training, validation, and testing
in .py / .mat etc in readable format with comments.
  - Stored model, associated weights or files (optional).
  - Any utils or assets or config. or checkpoints.

The submitted excel files will be evaluated using a semi-automated script to generate the relevant metrics against the true labels on the basis of which the participants will be ranked.

Following checklist will be used to select the top three
winning teams:
- Best evaluation metrics (especially mean AUC) on
testing dataset.
- In case of ties:
  * Model uniqueness and reproducibility.
  * Readability of the submitted report and github
repository.
  * Unique methods of handling class-imbalance
problems in the datasets.






















  
