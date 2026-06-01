# Diabetes_prediction



Here's a complete example of a Diabetes Prediction Model using Python with Machine Learning (Support Vector Machine) using the popular Pima Indians Diabetes Dataset.

1. Import Libraries
2. 
import numpy as np

import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler



from sklearn.svm import SVC




from sklearn.metrics import accuracy_score



3. Load Dataset


df = pd.read_csv("diabetes.csv")

print(df.head())



print(df.shape)



3. Separate Features and Target


X = df.drop('Outcome', axis=1)


Y = df['Outcome']



4. Split Data


X_train, X_test, Y_train, Y_test = train_test_split(


    X, Y,
    
    test_size=0.2,
    
    stratify=Y,
    
    random_state=42
    
)


5. Standardize Data



scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)



X_test = scaler.transform(X_test)


6. Train Model

7. 
model = SVC(kernel='linear')



model.fit(X_train, Y_train)



7. Evaluate Model


train_pred = model.predict(X_train)



train_acc = accuracy_score(Y_train, train_pred)




test_pred = model.predict(X_test)




test_acc = accuracy_score(Y_test, test_pred)





print("Training Accuracy:", train_acc)



print("Testing Accuracy:", test_acc)



8. Predict New Data

9. 
input_data = (6,148,72,35,0,33.6,0.627,50)





input_array = np.asarray(input_data)




input_array = input_array.reshape(1, -1)



input_scaled = scaler.transform(input_array)




prediction = model.predict(input_scaled)



if prediction[0] == 0:

    print("The person is NOT diabetic")

    
else:



    print("The person is diabetic")


    
Features in the Dataset


Feature	Description



Pregnancies	Number of pregnancies



Glucose	Glucose level



BloodPressure	Blood pressure


SkinThickness	Skin thickness


Insulin	Insulin level



BMI	Body Mass Index


DiabetesPedigreeFunction	Family history score


Age	Age of patient


Outcome	0 = Non-Diabetic, 1 = Diabetic


Other Algorithms You Can Try



LogisticRegression


DecisionTreeClassifier


RandomForestClassifier


KNeighborsClassifier



XGBClassifier

For this dataset, Random Forest and XGBoost often achieve higher accuracy than SVM.



create a readme file.



Diabetes Prediction using Machine Learning




Project Overview




This project predicts whether a person is diabetic or not using Machine Learning techniques.

The model is trained on the Pima Indians Diabetes Dataset and uses patient health information such as glucose level, blood pressure, BMI, age,


and other medical attributes to make predictions.

Dataset

The dataset contains medical records of patients and includes the following features:

Feature	Description


Pregnancies	Number of times pregnant


Glucose	Plasma glucose concentration



BloodPressure	Diastolic blood pressure (mm Hg)




SkinThickness	Triceps skin fold thickness (mm)


Insulin	2-Hour serum insulin (mu U/ml)



BMI	Body Mass Index



DiabetesPedigreeFunction	Diabetes pedigree function



Age	Age of the patient


Outcome	Target Variable (0 = Non-Diabetic, 1 = Diabetic)



Technologies Used


Python



NumPy



Pandas



Scikit-learn



Support Vector Machine (SVM)



Project Workflow


Load the Diabetes Dataset



Perform Data Preprocessing


Split the Dataset into Training and Testing Sets



Standardize the Feature Values



Train the SVM Model


Evaluate Model Performance


Build a Predictive System



Model Training



The model uses a Support Vector Machine (SVM) classifier with a linear kernel.



model = SVC(kernel='linear')


model.fit(X_train, Y_train)



Model Evaluation



The model performance is evaluated using Accuracy Score.




from sklearn.metrics import accuracy_score




train_accuracy = accuracy_score(Y_train, train_predictions)




test_accuracy = accuracy_score(Y_test, test_predictions)




Predictive System



The system accepts patient information as input and predicts whether the patient is diabetic or non-diabetic.



Example:



input_data = (6,148,72,35,0,33.6,0.627,50)




Output:




The person is diabetic




or

The person is NOT diabetic



Project Structure



Diabetes-Prediction/
│


├── diabetes.csv



├── Diabetes_Prediction.ipynb


├── diabetes_prediction.py


├── README.md



└── requirements.txt



Installation



