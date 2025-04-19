# 🧠 AKI Prediction System

A Flask-based web application to predict renal recovery outcomes in ICU patients diagnosed with Acute Kidney Injury (AKI) using machine learning.

## 📌 Description

This system uses patient vitals, lab results, and clinical data to determine the likelihood of recovery or death. The prediction is powered by a Random Forest Classifier and presented via a responsive web UI. Results are stored in a SQLite database, and model performance can be visualized through an interactive dashboard.

---

## 🚀 Features

- 📥 Patient data input via web form  
- 🔮 Real-time AKI recovery predictions  
- 📊 Model performance comparison (Random Forest & Logistic Regression)  
- 🧾 View previously predicted patient data  
- 🧠 Trained on real ICU patient dataset  

---

## 🛠️ Tech Stack

- **Frontend**: HTML, Tailwind CSS, Chart.js  
- **Backend**: Python, Flask  
- **ML Models**: scikit-learn (Random Forest, Logistic Regression)  
- **Database**: SQLite  

---

## 📂 Project Structure

```
├── app.py                      # Main Flask backend  
├── aki_data.db                 # SQLite database storing patient data  
├── AKIrequired_dataset.csv     # Dataset used for training  
├── templates/  
│   ├── index.html              # Input form page  
│   ├── result.html             # Prediction result display  
│   ├── performance.html        # Model performance dashboard  
│   └── previous_data.html      # Historical prediction data  
```

---

## 🧪 How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/your-username/aki-prediction-system.git
cd aki-prediction-system
```

2. **Install dependencies**
```bash
pip install flask pandas numpy scikit-learn
```

3. **Run the app**
```bash
python app.py
```

4. **Access it in your browser**
```
http://127.0.0.1:5000/
```

---

## 📈 Sample Model Accuracy

| Model               | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Random Forest       | 83.5%    | Varies  |
| Logistic Regression | 80.0%    | 89.0%   |

---

## 📬 Contact

For questions or suggestions, feel free to open an issue or reach out.

---

> ⚠️ This tool is for educational and research purposes only. It is not intended for clinical use.
