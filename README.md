# 💼 Employee Salary Predictor App

This project is a machine learning-based web app developed using **Streamlit** that predicts the estimated **monthly** and **yearly** salary of an employee based on personal and professional attributes.



## 📌 Features

- 🔍 Predict salary using a trained **Random Forest Regressor**
- 📊 Visual output with bar chart
- 🧠 Uses both ML and fallback rule-based estimation
- 📱 Mobile-friendly responsive design
- 🎨 Clean UI with emojis and modern styling
- 🖼️ Includes evaluation images (model accuracy and importance)

## 🧠 Inputs Used

The app takes the following inputs:

- 📅 Age  
- ⚧️ Gender  
- 🎓 Education Level  
- 💼 Job Title  
- 🧪 Years of Experience  

## 📊 Technologies Used

- **Python 3.13+**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **Pickle**
- **Plotly**
- **Seaborn / Matplotlib** (for model evaluation charts)

## 🔧 Setup Instructions


### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

`streamlit run app.py

## 📈 Model Evaluation

* **Model Used**: Random Forest Regressor
* **R² Score**: 87.52%
* **MAE**: ₹3,200 (approx)

The app includes evaluation charts that visualize model accuracy and feature importance.

## 🌐 Live App

Try the live deployed version here:
🔗 https://employe-salary-predictor-app-dvhqvuv8n89ukgqgaxkox5.streamlit.app/

(*Replace with your real link after deployment*)

## 📝 References

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011)Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.https://scikit-learn.org
Waskom, M. L. (2021)Seaborn: Statistical Data Visualization. Journal of Open Source Software, 6(60), 3021.https://seaborn.pydata.org
Streamlit Inc. (2023)Streamlit: Turn Python Scripts into Interactive Web Apps.https://streamlit.io
Employee Salary Dataset – Used from internal project resources / publicly available dataset Kaggle https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer
<img width="3454" height="660" alt="image" src="https://github.com/user-attachments/assets/553d5316-de0a-46ef-b169-f85e8a98bdb7" />

## 🙋 About Me

**👩 Risha Gupta**
📚 MCA Student | Lovely Professional University
📫 Email: [rishagupta463@egmail.com]
🛠️ Passionate about ML and Web Apps

## 💡 Future Scope

* Add support for real-time API-based data
* Use more complex deep learning models
* Provide downloadable salary report
* Enable saving past user inputs
* Add authentication for multiple users


