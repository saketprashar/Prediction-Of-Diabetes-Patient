# Prediction-Of-Diabetes-Patient
This project uses machine learning models to predict whether a person is likely to have diabetes, based on health and medical data. It uses the Pima Indians Diabetes Dataset, a standard dataset widely used for binary classification problems in healthcare.
# Objective
To develop a predictive model that can accurately classify patients as diabetic or non-diabetic based on features like glucose level, BMI, age, blood pressure, and more — helping in early diagnosis and prevention.
# Features
•	Data preprocessing and cleaning
•	Exploratory Data Analysis (EDA)
•	Feature scaling and selection
•	Model training and evaluation
•	Accuracy, confusion matrix, precision & recall reports
•	Predictive system for new patient data
# Technologies & Libraries
*Tool/Libraries        * Purpose
Python              -  Programming language
Pandas, NumPy       -  Data handling
Matplotlib, Seaborn -  Data visualization
Scikit-learn        -  ML algorithms and evaluation
Jupyter Notebook    -  Interactive development
Streamlit (optional)-  Web app interface
# Machine Learning Models Used
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- k-Nearest Neighbors (KNN)
All models are evaluated using metrics like *accuracy, **precision, **recall, **F1-score, and **confusion matrix*.
# Project Structure
Diabetes_Prediction/
├── data/
│   └── diabetes.csv             # Pima dataset
├── notebooks/
│   └── model_training.ipynb     # ML model notebook
├── models/
│   └── diabetes_model.pkl       # Saved ML model
├── app/
│   └── diabetes_predictor.py    # (Optional) Streamlit app
├── requirements.txt
└── README.md
# Dataset information
Dataset: Pima Indians Diabetes Database
	•	Source: UCI Machine Learning Repository / Kaggle
	•	Features:
	•	Pregnancies
	•	Glucose
	•	Blood Pressure
	•	Skin Thickness
	•	Insulin
	•	BMI
	•	Diabetes Pedigree Function
	•	Age
	•	Outcome (0 = No Diabetes, 1 = Diabetes)
 ---

# How to Run the Project
1. *Clone the repository*  
   ```bash
   git clone https://github.com/yourusername/Diabetes_Prediction.git
   cd Diabetes_Prediction
# Install the dependencies
pip install -r requirements.txt
# Run the Jupyter Notebook
jupyter notebook notebooks/model_training.ipynb
# (Optional) Run the Streamlit App
streamlit run app/diabetes_predictor.py
# Model Accuracy (Example Results)
Model               Accuracy
Logistic Regression 78%
Random Forest       84%
SVM                 81%

# Future Enhancements
	•	Add deep learning models (e.g., ANN)
	•	Integrate real-time input form
	•	Model interpretability with SHAP or LIME
	•	Deploy app on Streamlit Cloud or Heroku
# Contributions
Contributions, improvements, and ideas are always welcome.
Feel free to fork the repo, make changes, and submit a pull request.
# License
This project is licensed under the MIT License.
# Links
	•	Dataset: Kaggle - Pima Diabetes
	•	Author: Saket Prashar

 
