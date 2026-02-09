# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Streamlit UI Setup
# st.title("Efficient Online Fraud Detection for Healthcare Transactions")
# st.write("Upload a healthcare transactions dataset to train a model or enter transaction details for prediction.")

# # File uploader
# dataset = st.file_uploader("Upload CSV file", type=["csv"])

# if dataset is not None:
#     df = pd.read_csv(dataset)
#     st.write("### Dataset Preview:", df.head())

#     # Fill missing values
#     df.fillna(0, inplace=True)

#     # Encoding categorical variables
#     le = LabelEncoder()
#     df['type'] = le.fit_transform(df['type'])
#     df['nameOrig'] = le.fit_transform(df['nameOrig'])
#     df['nameDest'] = le.fit_transform(df['nameDest'])

#     # Splitting Data
#     X = df.drop('isFraud', axis=1)
#     y = df['isFraud']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Model Selection
#     model_option = st.selectbox("Select Model", ["Random Forest", "KNN", "AdaBoost"])

#     if st.button("Train Model"):
#         if model_option == "Random Forest":
#             model = RandomForestClassifier(n_estimators=100)
#         elif model_option == "KNN":
#             model = KNeighborsClassifier(n_neighbors=7)
#         else:
#             model = AdaBoostClassifier()

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         # Model Evaluation
#         acc = accuracy_score(y_test, y_pred)
#         st.write(f"### Model Accuracy: {acc:.2f}")
#         st.write("### Classification Report:")
#         st.text(classification_report(y_test, y_pred))

#         # Confusion Matrix
#         cm = confusion_matrix(y_test, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#         st.pyplot(fig)

#         # Save Model
#         joblib.dump(model, "fraud_detection_model.pkl")

# # Prediction Section
# st.write("## Predict a Healthcare Transaction")
# try:
#     model = joblib.load("fraud_detection_model.pkl")
#     input_features = []
#     feature_names = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
    
#     for feature in feature_names:
#         value = st.number_input(f"Enter {feature}", min_value=0.0)
#         input_features.append(value)
    
#     if st.button("Predict"):
#         prediction = model.predict([input_features])
#         result = "Fraudulent Transaction" if prediction[0] == 1 else "Genuine Transaction"
#         st.write(f"### Prediction: {result}")
# except:
#     st.write("Please train a model first before making predictions.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit UI Setup
st.title("Efficient Online Fraud Detection System for Healthcare Transactions")
st.write("Upload a dataset to train a model or enter transaction details for prediction.")

# File uploader
dataset = st.file_uploader("Upload CSV file", type=["csv"])

if dataset is not None:
    data = pd.read_csv(dataset)
    st.write("### Dataset Preview:", data.head())

    # Data Preprocessing
    data.fillna(0, inplace=True)
    le = LabelEncoder()
    data['type'] = le.fit_transform(data['type'])
    data['nameOrig'] = le.fit_transform(data['nameOrig'])
    data['nameDest'] = le.fit_transform(data['nameDest'])
    
    X = data.drop('isFraud', axis=1)
    y = data['isFraud']
    
    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_option = st.selectbox("Select Model", ["Random Forest", "KNN", "AdaBoost"])
    
    if st.button("Train Model"):
        if model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_option == "KNN":
            model = KNeighborsClassifier(n_neighbors=7)
        else:
            model = AdaBoostClassifier()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Evaluation
        acc = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {acc:.2f}")
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Save Model
        joblib.dump(model, "fraud_detection_model.pkl")
        joblib.dump(le, "label_encoder.pkl")

st.write("## Predict a Transaction")

# Load trained model and encoder
try:
    model = joblib.load("fraud_detection_model.pkl")
    le = joblib.load("label_encoder.pkl")

    step = st.number_input('Step', min_value=0)
    trans_type = st.text_input('Transaction Type')
    amount = st.number_input('Amount', min_value=0.0)
    nameOrig = st.text_input('Name Origin')
    oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0)
    newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0)
    nameDest = st.text_input('Name Destination')
    oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0)
    newbalanceDest = st.number_input('New Balance Destination', min_value=0.0)
    isFlaggedFraud = st.number_input('Is Flagged Fraud', min_value=0, max_value=1)

    if st.button("Predict"):
        input_data = pd.DataFrame([[step, trans_type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                                    nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud]],
                                  columns=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
                                           "nameDest", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"])
        
        input_data['type'] = le.transform(input_data['type'])
        input_data['nameOrig'] = le.transform(input_data['nameOrig'])
        input_data['nameDest'] = le.transform(input_data['nameDest'])
        
        prediction = model.predict(input_data)
        result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"
        st.write(f"### Prediction: {result}")

except Exception as e:
    st.write("Please train a model first before making predictions.")