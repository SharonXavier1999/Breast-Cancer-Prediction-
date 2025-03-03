# Breast Cancer Prediction System

## 📌 Project Overview
This **Breast Cancer Prediction System** is a machine learning-based web application designed to assist in the early detection of breast cancer. The system predicts whether a tumor is **benign** or **malignant** based on cell nuclei measurements from **fine needle aspiration (FNA) biopsy samples**. It features **risk level indicators, a tumor severity gauge, and a doctor consultation recommendation**, making the predictions more interpretable and user-friendly.

## ✨ Features
- ✅ **Machine Learning-Based Prediction** (Logistic Regression, SVM, Random Forest, XGBoost)
- ✅ **Interactive Web App with Streamlit**
- ✅ **Risk Level Indicator (Low, Moderate, High)** for better decision-making
- ✅ **Tumor Severity Gauge** for visualizing cancer risk
- ✅ **Doctor Consultation Recommendation** for immediate action
- ✅ **SHAP Explainability** to show which features influenced the prediction
- ✅ **Real-time Radar Chart** for tumor feature visualization
- ✅ **SMOTE for Handling Class Imbalance** (Synthetic Minority Over-sampling Technique)
- ✅ **PCA Visualization** to understand tumor separability


## 🚀 Usage Instructions
1. **Open the Web App** in your browser after running Streamlit.
2. **Adjust the sliders** to input tumor cell measurements.
3. **View Prediction**: The model will classify the tumor as **Benign (0)** or **Malignant (1)**.
4. **Interpret Results**:
   - **Risk Level Indicator** shows **Low, Moderate, or High risk**.
   - **Tumor Severity Gauge** provides a **visual risk assessment**.
   - **Doctor Consultation Recommendation** suggests medical action if necessary.
   - **SHAP Visualizations** help understand which features influenced the model's decision.

## 🔬 Model Performance & Results
- **Best Model:** **XGBoost**
- **Accuracy:** **98.1%**
- **Precision:** **98%** (Malignant), **97%** (Benign)
- **Recall:** **95%** (Malignant), **99%** (Benign)
- **AUC-ROC Score:** **0.99** (Excellent model performance)
- **Confusion Matrix Summary:**
  - ✅ **Correct Predictions:** 69 Benign, 42 Malignant
  - ❌ **False Positives:** 2 (Benign misclassified as Malignant)
  - ❌ **False Negatives:** 1 (Malignant misclassified as Benign) → Critical for improvement!

## 📊 Visualizations
- **PCA Visualization** to show tumor separation
- **ROC Curve** to compare model performance
- **Radar Chart** to visualize tumor features
- **SHAP Summary & Waterfall Plots** for explainability

## 🔮 Future Enhancements
- 📌 **Deep Learning Integration (CNNs) for Histopathology Images**
- 📌 **Real-time Data Collection from Cytology Labs**
- 📌 **Mobile App Version for Quick Diagnosis**
- 📌 **Multilingual Support for Wider Accessibility**
- 📌 **Integration with Electronic Health Records (EHR)**


