import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image

st.set_page_config(layout="wide")

image = Image.open('Banner.jpg')
st.image(image)


def load_data():
    data = pd.read_csv("Diabetes_Dataset.csv")
    data = data[data['gender'] != 'Other']  # Remove rows with 'Other' in gender
    return data

def preprocess_trainmodel(data, normalization, classifier_choice):
    # Define categorical and numerical columns
    cat_cols = ['gender']
    num_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    # Choose scaler based on user selection
    if normalization == 'Z-Score Normalization':
        norm = StandardScaler()
    elif normalization == 'No Normalization':
        norm = 'passthrough'
    else:
        norm = MinMaxScaler()

    # Choose classifier based on user selection
    if classifier_choice == 'Random Forest':
        classifier = RandomForestClassifier()
    elif classifier_choice == 'Adaboost':
        classifier = AdaBoostClassifier()
    elif classifier_choice == 'SVM':
        classifier = SVC(probability=True)
    else:
        classifier = DecisionTreeClassifier()
    
     # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_cols),
            ('cat', OneHotEncoder(categories=[['Male', 'Female']]), cat_cols)
        ])
    
    # Append classifier to preprocessing pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    
    # Separate the features and the target variable
    X = data[num_cols + cat_cols]
    y = data['diabetes']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Predict probabilities for the positive class
    y_score = clf.predict_proba(X_test)[:, 1]

    return clf, accuracy, normalization, y_test, y_score

def plot_precision_recall_curve(y_test, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    average_precision = average_precision_score(y_test, y_scores)

    # Plotting Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    ax.plot(recall, precision, color='blue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    plt.grid(False)

    return fig

def predict_diabetes(model, input_data):
    # Predict the outcome for input data
    prediction = model.predict(input_data)
    return prediction

def main():
    st.markdown("<h1 style='text-align: center;'>Diabetes Prediction Website</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This website predicts whether a person has diabetes or not depending on their health condition!</p>", unsafe_allow_html=True)

    # Load data
    data = load_data()

    # Create two columns for the selectboxes
    col1, col2 = st.columns(2)
    # Place a selectbox in each column
    with col1:
        normalization = st.selectbox('Select Normalization Method', ['No Normalization', 'MinMax Normalization', 'Z-Score Normalization'])
    with col2:
        classifier = st.selectbox('Select Classifier', ['Decision Tree', 'Random Forest', 'AdaBoost', 'SVM'])

    model, model_accuracy, norm_used, y_test, y_scores = preprocess_trainmodel(data, normalization, classifier)

    # Inject custom CSS to style the button
    st.markdown("""
    <style>
    /* General style for all buttons */
    button {
        font-size: 20px;  /* Increases the font size */
        height: 3em;      /* Increases the height */
        width: 100%;      /* Makes the button fill the column width */
        margin: 1em 0;    /* Adds some vertical spacing */
    }
    </style>
    """, unsafe_allow_html=True)

    # Create two columns with different widths
    button_col, graph_col, extra_col = st.columns([1, 2, 1])

    # Button in the left column
    with button_col:
        if st.button('Click Me'):
            model_trained = True  # You can use this flag to control graph rendering
        else:
            model_trained = False

    # Graph in the right column
    with graph_col:
        if model_trained:
            # Assuming you have these functions defined and they operate as expected
            model, model_accuracy, norm_used, y_test, y_scores = preprocess_trainmodel(data, normalization, classifier)

            st.header(f"Model Accuracy: {model_accuracy:.2%}")  # Formats the accuracy as a percentage

            fig = plot_precision_recall_curve(y_test, y_scores)
            st.pyplot(fig)
        else:
            st.write("Click the button to the left to find the Accuracy Score and to see the Precision-Recall Curve plot!")

    # Creating form for user input
    with st.form(key='diabetes_form'):
        age = st.slider('Age', min_value=1, max_value=80, value=30)
        hypertension = st.selectbox('Hypertension (0 for No, 1 for Yes)', [0, 1])
        heart_disease = st.selectbox('Heart Disease (0 for No, 1 for Yes)', [0, 1])
        bmi = st.slider('BMI', min_value=11.0, max_value=95.69, value=20.0)
        hba1c_level = st.slider('HbA1c Level', min_value=3.0, max_value=9.0, value=4.5)
        blood_glucose_level = st.slider('Blood Glucose Level', min_value=80, max_value=300, value=90)
        gender = st.selectbox('Gender', ['Male', 'Female'])

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Store inputs into dataframe
        input_dict = {'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease, 'bmi': bmi,
                      'HbA1c_level': hba1c_level, 'blood_glucose_level': blood_glucose_level, 
                      'gender': gender}
        input_df = pd.DataFrame([input_dict])
        
        # Get prediction
        prediction = predict_diabetes(model, input_df)
        if prediction[0] == 1:
            st.success('The prediction is: Diabetes')
        else:
            st.success('The prediction is: No Diabetes')

if __name__ == '__main__':
    main()
