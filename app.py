from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


DATABASE_PATH = os.path.abspath("aki_data.db")  

# Initialize Database
def init_db():
    conn = sqlite3.connect('aki_data.db')
    c = conn.cursor()

    # Correct table name: Ensure you're using "patient_data" consistently
    c.execute('''CREATE TABLE IF NOT EXISTS patient_data (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  age REAL, gender TEXT, bic_max REAL, bic_mean REAL, bic_min REAL,
                  bilirubin REAL, bp_max REAL, bp_mean REAL, bp_min REAL,
                  bun_max REAL, bun_mean REAL, bun_min REAL, days_in_uci INTEGER,
                  fio2 REAL, gcs_max REAL, gcs_mean REAL, gcs_min REAL,
                  hr_max REAL, hr_mean REAL, hr_min REAL, max_pao2 REAL,
                  mean_pao2 REAL, min_pao2 REAL, pot_max REAL, pot_mean REAL,
                  pot_min REAL, sod_max REAL, sod_mean REAL, sod_min REAL,
                  temp REAL, wbc_max REAL, wbc_mean REAL, wbc_min REAL,
                  aki INTEGER
    )''')

    conn.commit()

    # Debugging: Print tables to confirm creation
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    print("Tables in database:", tables)  # Should print patient_data

    conn.close()

# Run initialization
init_db()

# Load and preprocess dataset
def load_and_preprocess_data():
    try:
        df = pd.read_csv("AKIrequired_dataset - final.csv")
    except FileNotFoundError:
        print("The file 'AKIrequired_dataset - final.csv' was not found. Please ensure it is in the same directory as the script.")
        exit()

    # Encode categorical variables (if 'gender' exists in the dataset)
    if 'gender' in df.columns:
        encoder = LabelEncoder()
        df['gender'] = encoder.fit_transform(df['gender'])
    
    df.fillna(df.mean(), inplace=True)  # Replace NaNs with column means

    X = df.drop(columns=["IHM"])  # Use all features except IHM
    y = df["IHM"]  # Target variable

    print("Dataset Loaded - Features:", X.shape[1], " Target:", y.shape[0])  # Debugging

    return X, y

# Train model
def train_and_evaluate_model():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, accuracy, roc_auc, report

# Fetch previous data from the database
def get_previous_data():
    conn = sqlite3.connect('aki_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM patient_data ORDER BY id DESC")
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    conn.close()
    
    if not rows:
        return None
    
    # Convert database rows to list of dictionaries, ensuring all columns are filled
    return [dict(zip(columns, row)) for row in rows]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data from the frontend
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided', 'success': False}), 400

        print("Received data:", data)  # Debugging

        # Convert gender from string ('male', 'female') to numerical values (1, 0, 2)
        gender_map = {"male": 1, "female": 0, "other": 2}
        if 'gender' in data:
            data['gender'] = gender_map.get(data['gender'].lower(), 2)  # Default to 2 for unknown genders

        # Convert dictionary values to a list for model input
        try:
            input_values = [float(data[key]) if key in data and data[key] != "" else 0.0 for key in data]
        except ValueError as ve:
            return jsonify({'error': f'Invalid input format: {str(ve)}', 'success': False}), 400

        print("Transformed input values:", input_values)  # Debugging

        # Load trained model
        model, _, _, _ = train_and_evaluate_model()

        # Validate input length against model's expected input shape
        expected_features = model.n_features_in_
        received_features = len(input_values)

        print(f"Expected input features: {expected_features}, Received: {received_features}")

        if received_features != expected_features:
            error_message = f"Model expects {expected_features} features, but received {received_features}."
            print("Error:", error_message)
            return jsonify({'error': error_message, 'success': False}), 400

        # Reshape input for model prediction
        prediction = model.predict(np.array(input_values).reshape(1, -1))[0]

        # Interpret result
        result_text = "Patient will be recovered" if prediction == 0 else "Patient has chances to die"
        print("Prediction successful:", result_text)  # Debugging

        # Store patient data in SQLite
        try:
            conn = sqlite3.connect('aki_data.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO patient_data (
                    age, gender, bic_max, bic_mean, bic_min, bilirubin, bp_max, bp_mean, bp_min, 
                    bun_max, bun_mean, bun_min, days_in_uci, fio2, gcs_max, gcs_mean, gcs_min, 
                    hr_max, hr_mean, hr_min, max_pao2, mean_pao2, min_pao2, pot_max, pot_mean, pot_min, 
                    sod_max, sod_mean, sod_min, temp, wbc_max, wbc_mean, wbc_min, aki
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(input_values) + (prediction,))
            conn.commit()
        except sqlite3.Error as db_error:
            print("Database Insertion Error:", str(db_error))
            return jsonify({'error': f'Database error: {str(db_error)}', 'success': False}), 500
        finally:
            conn.close()

        # Return JSON response with redirect URL for frontend
        return jsonify({
            'success': True,
            'prediction': result_text,
            'redirect_url': '/result'
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'error': f'Prediction failed: {str(e)}', 'success': False}), 500
    
@app.route('/performance')
def performance():
    _, accuracy, roc_auc, report = train_and_evaluate_model()
    return render_template("performance.html", accuracy=accuracy, roc_auc=roc_auc, class_report=report)

@app.route('/previous_data')
def previous_data():
    conn = sqlite3.connect("aki_data.db")
    cursor = conn.cursor()
    
    # Fetch last 10 records (Modify query as needed)
    cursor.execute("SELECT * FROM patient_data ORDER BY id DESC LIMIT 10;")
    data = cursor.fetchall()

    # Extract column names
    columns = [desc[0] for desc in cursor.description]

    conn.close()

    # Process data to handle binary `aki` field
    processed_data = []
    for row in data:
        row = list(row)
        # Convert `aki` binary data to an integer (0 or 1)
        aki_value = int.from_bytes(row[-1], byteorder="little") if isinstance(row[-1], bytes) else row[-1]
        row[-1] = "PATIENT HAS CHANCES TO DIE" if aki_value == 1 else "PATIENT WILL BE RECOVERED"
        processed_data.append(row)

    # Debug print
    print("Processed Data:", processed_data)

    return render_template("previous_data.html", data=processed_data, columns=columns)

if __name__ == "__main__":
    app.run(debug=True)