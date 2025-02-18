# Loan Status Prediction

## Project Overview
This project focuses on predicting loan approval status based on various applicant details using machine learning techniques. The dataset contains information about applicants, such as income, loan amount, credit history, and other factors, which are used to train a classification model.

## Dataset
Data is uploaded here.
The dataset (`Loan_data.csv`) contains multiple features, including:
- **Applicant Income**
- **Coapplicant Income**
- **Loan Amount**
- **Loan Amount Term**
- **Credit History**
- **Gender, Married, Dependents, Education, Self_Employed** (Categorical variables)
- **Loan Status (Target Variable)**

## Dependencies
The following Python libraries are required to run the notebook:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

## Installation
Ensure you have Python installed along with the required libraries. You can install them using:
```sh
pip install numpy pandas scikit-learn
```

## Data Processing
1. Import the dataset and Load the dataset using Pandas.
2. Perform data cleaning by handling missing values.
3. Convert categorical variables into numerical values.
4. Split the dataset into training and testing sets.

## Model Training
- The Support Vector Machine (SVM) model is used for classification.
- The dataset is split into training and testing sets using an 90-10 ratio.
- The model is trained on the training dataset.

## Model Evaluation
- The accuracy of the trained model is calculated using the test dataset. **(75-85%)**
- The accuracy score is printed as output.

## How to Run
1. Open the `Loan_Status_Prediction.ipynb` file in Jupyter Notebook or Google Colab.
2. Execute the cells sequentially to process the data and train the model.
3. Observe the accuracy score of the model.

## Future Enhancements
- Implement additional machine learning models (e.g., Decision Tree, Random Forest) for comparison.
- Improve feature engineering for better predictions.
- Tune hyperparameters for better accuracy.

## Contribution
Feel free to contribute by opening an issue or submitting a pull request.

## License
This project is licensed under the MIT License.
