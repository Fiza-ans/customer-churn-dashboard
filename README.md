# customer-churn-dashboard
This README explains how to run the Customer Churn Prediction Streamlit application on your system.
---
## 1️ Requirements
Make sure the following are installed:
- Python 3.10 or above
- pip (Python package manager)
- Internet connection (for installing libraries)
Check Python version:

## 2 Download the Project
Download or clone the project folder:
git clone https://github.com/your-username/customer-churn-dashboard.git

## 3️ Install Required Libraries
Install all dependencies using:
pip install -r requirements.txt
This will install:
- streamlit
- pandas
- scikit-learn
- xgboost
- matplotlib
- plotly
- joblib

## 4️ Project Files Required
Ensure these files exist inside the folder:
app.py
final_model.pkl
requirements.txt

## 5️ Run the Application
Start the Streamlit dashboard: streamlit run app.py

## 6️ Open the Dashboard
After running the command, a browser window will open automatically.
If not, open manually: http://localhost:8501

## 7️ How to Use
1. Enter customer details in the form.
2. Click **Predict Churn**.
3. View:
   - Churn prediction
   - Probability score
   - Risk level
   - Feature importance
   - Retention recommendations

## 8️ Stop the Application
Press: CTRL + C
in the terminal to stop the server.

## Author
Fiza Ansari





