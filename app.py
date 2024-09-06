from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the entire model pipeline
model_file_path = 'churn_prediction_model1.pkl'
numeric_values_file_path = 'numeric_unique_values.pkl'
categorical_values_file_path = 'categorical_unique_values.pkl'

with open(model_file_path, 'rb') as f:
    model_pipeline = pickle.load(f)  # Load the entire pipeline

# Extract the preprocessor and the model from the pipeline
preprocessor = model_pipeline.named_steps['preprocessor']
scaler = model_pipeline.named_steps['scaler']
model = model_pipeline.named_steps['classifier']

# Load unique values
with open(numeric_values_file_path, 'rb') as f:
    numeric_unique_values = pickle.load(f)

with open(categorical_values_file_path, 'rb') as f:
    categorical_unique_values = pickle.load(f)

# Define features for prediction
features_for_prediction = {
    'numeric': [col for col in numeric_unique_values.keys() if col != 'Churn'],
    'categorical': [col for col in categorical_unique_values.keys() if col != 'Churn']
}

numeric_features = features_for_prediction['numeric']
categorical_features = features_for_prediction['categorical']

@app.route('/')
def index():
    dropdowns = {
        'numeric': {col: sorted(list(values)) for col, values in numeric_unique_values.items() if col != 'Churn'},
        'categorical': {col: sorted(list(values)) for col, values in categorical_unique_values.items() if col != 'Churn'}
    }
    return render_template('index.html', dropdowns=dropdowns, features=features_for_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    inputs = {col: request.form.get(col) for col in features_for_prediction['numeric']}
    inputs.update({col: request.form.get(col) for col in features_for_prediction['categorical']})

    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs])

    # Preprocess the data using the pipeline
    preprocessed_data = preprocessor.transform(input_df)
    scaled_data = scaler.transform(preprocessed_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    return render_template('result.html', prediction=prediction)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)

                # Ensure the dataframe has the necessary columns including CustomerID
                columns_needed = ['CustomerID'] + features_for_prediction['numeric'] + features_for_prediction[
                    'categorical']
                if not all(col in df.columns for col in columns_needed):
                    return render_template('upload.html', error_message="Error: CSV file missing required columns.")

                # Preprocess the data
                df = df[columns_needed]

                # Handle missing values and convert data types
                df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')
                df[numeric_features].fillna(df[numeric_features].mean(), inplace=True)
                df[categorical_features] = df[categorical_features].astype(str)
                df[categorical_features].fillna('Unknown', inplace=True)

                # Apply preprocessing and scaling
                preprocessed_data = preprocessor.transform(df.drop(columns=['CustomerID']))
                scaled_data = scaler.transform(preprocessed_data)

                # Predict
                predictions = model.predict(scaled_data)
                df['Prediction'] = predictions

                # Identify customers predicted to churn
                churn_customers = df[df['Prediction'] == 1]

                # Limit to first 6 columns (including 'Prediction')
                columns_to_display = df.columns[:6].tolist() + ['Prediction']
                df = df[columns_to_display]

                # Count churn and no churn
                churn_count = df['Prediction'].sum()
                no_churn_count = len(df) - churn_count

                # Convert DataFrame to HTML table
                table_html = df.to_html(classes='table table-striped table-bordered', index=False)

                # Convert churn customers to HTML table
                if not churn_customers.empty:
                    churn_table_html = churn_customers[['CustomerID', 'Prediction']].to_html(
                        classes='table table-striped table-bordered', index=False)
                    churn_ids = ', '.join(map(str, churn_customers['CustomerID'].tolist()))
                else:
                    churn_table_html = ""
                    churn_ids = ""

                # Create user-friendly messages
                if churn_count > no_churn_count:
                    message = (
                        f"Out of {len(df)} customers, {churn_count} are predicted to churn (not continue with the business) "
                        f"and {no_churn_count} are predicted to continue with the business. "
                        "Consider implementing retention strategies for the customers at risk."
                    )
                else:
                    message = (
                        f"Out of {len(df)} customers, {churn_count} are predicted to churn (not continue with the business) "
                        f"and {no_churn_count} are predicted to continue with the business. "
                        "The majority of customers are predicted to stay, which is a positive sign."
                    )

                churn_message = (
                    f"Customers predicted to churn (CustomerID): {churn_ids}. "
                    "These customers should be targeted for enhanced offers to improve retention."
                )

                return render_template('csv_result.html', table_html=table_html, message=message,
                                       churn_table_html=churn_table_html, churn_message=churn_message)
            except Exception as e:
                return f"Error during prediction: {e}", 500

    return render_template('upload.html')


@app.route('/document')
def document():
    return render_template('document.html')


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
