# AI-Based E-Waste Identification Using Camera Input

## Module
M516 – Business Project in Big Data & AI

## Project Overview
Electronic waste (e-waste) is one of the fastest growing waste streams worldwide, yet recycling rates remain low due to limited automation in waste identification and sorting processes.

This project presents a proof-of-concept AI system that uses computer vision to identify types of e-waste from camera input. The system is designed to run locally and demonstrates how image classification models can support sustainability-focused decision making in recycling and waste management contexts.

The project focuses on feasibility, interpretability, and reproducibility rather than production-scale deployment.

---

## Objectives
The key objectives of this project are:

1. To analyse and understand publicly available e-waste image datasets.
2. To perform exploratory data analysis (EDA) to assess data quality and suitability.
3. To train and evaluate an image classification model for e-waste identification.
4. To deploy the trained model locally and demonstrate real-time inference using a webcam.
5. To critically evaluate the model’s performance and limitations in a sustainability context.

---

## Project Structure
eco-vision-e-waste/
│
├── notebooks/
│ ├── 01_problem_and_data_understanding.ipynb
│ ├── 02_exploratory_data_analysis.ipynb
│ ├── 03_model_training_and_evaluation.ipynb
│ ├── 04_local_webcam_inference.ipynb
│
├── src/
│ ├── model_utils.py
│ ├── webcam_utils.py
│
├── outputs/
│ ├── figures/
│ ├── metrics/
│
├── requirements.txt
├── .gitignore
└── README.md


---

## Datasets

### 1. Kaggle E-Waste Image Dataset (Training & Validation)
Source:  
https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset

This dataset provides labelled images of common electronic waste categories and is used as the primary dataset for model training and evaluation.

### 2. Roboflow E-Waste Dataset (Inference & Demo)
Source:  
https://universe.roboflow.com/electronic-waste-detection/e-waste-dataset-r0ojc

This dataset contains more diverse real-world images and is used for qualitative testing and webcam inference demonstrations. It is not used during training to avoid data leakage and annotation inconsistencies.

---

## Methodology Summary

1. **Problem Definition & Data Understanding**  
   Define the sustainability challenge and analyse the available datasets.

2. **Exploratory Data Analysis (EDA)**  
   Examine class distribution, image resolution, data quality, and preprocessing needs.

3. **Model Training & Evaluation**  
   Train a convolutional neural network using transfer learning and evaluate its performance using standard classification metrics.

4. **Local Webcam Inference**  
   Deploy the trained model locally and perform real-time classification using webcam input via OpenCV.

---

## Local Webcam Demonstration

To meet the project requirement of running the model on localhost, the trained model is executed on a local machine using Python and OpenCV.

The webcam demo:
- Captures live frames from the local camera
- Preprocesses frames to match the training pipeline
- Displays the predicted e-waste category and confidence score in real time

This demonstration is recorded as part of the final video submission.

---

## How to Run the Project

### 1. Training and Analysis (Google Colab)
- Run notebooks `01` to `03` in Google Colab.
- Download the trained model after completion.

### 2. Local Inference (Local Machine)

Install dependencies:
```bash
pip install -r requirements.txt

jupyter notebook notebooks/04_local_webcam_inference.ipynb

