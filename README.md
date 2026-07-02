# HR Employee Attrition Prediction System

## 📋 Overview

A machine learning-based system to predict employee attrition risk, helping HR professionals make data-driven retention decisions.

**Live Demo**: [Your Streamlit App URL]

---

## 🎯 Business Impact

- **Reduce turnover costs** by 20-30%
- **Improve employee retention** rates
- **Data-driven HR decisions**
- **Proactive intervention** for at-risk employees

---

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| **Employees** | 1,470 records |
| **Features** | 35 attributes |
| **Target** | Attrition (Yes/No) |
| **Attrition Rate** | 16.12% |

### Key Features Used

| Category | Features |
|----------|----------|
| **Personal** | Age, Gender, Marital Status, Education |
| **Job** | Department, Role, Level, OverTime, Travel |
| **Compensation** | Monthly Income, Salary Hike, Performance |
| **Satisfaction** | Job, Environment, Relationship, Work-Life |
| **Tenure** | Years at Company, Promotion History |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hr-attrition-prediction.git
cd hr-attrition-prediction

# Install dependencies
pip install -r requirements.txt

# Train model
python model_training.py

# Run application
streamlit run app.py
