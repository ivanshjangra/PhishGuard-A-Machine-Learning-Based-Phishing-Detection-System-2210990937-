# PhishGuard

### A Machine Learning-Based Phishing Detection System

PhishGuard is a defensive cybersecurity research project designed to detect phishing URLs using machine learning, feature engineering, and explainable AI techniques. The system combines lexical, structural, and semantic URL analysis with ensemble learning models to classify malicious and legitimate URLs in real time.

The framework also integrates SHAP (SHapley Additive exPlanations) to provide transparent and interpretable phishing predictions suitable for analyst visibility and SOC-oriented deployment workflows.

---

# Features

* Multi-feature phishing URL detection
* Lexical, structural, and semantic analysis
* Ensemble machine learning classification
* Real-time phishing prediction
* SHAP-based explainability
* Suspicious TLD and IP detection
* Brand impersonation analysis
* Streamlit-based dashboard
* Low-latency prediction pipeline
* SOC deployment-oriented architecture

---

# Technology Stack

## Backend / Machine Learning

* Python 3.11
* Scikit-learn
* XGBoost
* SHAP

## Data Processing

* Pandas
* NumPy
* SciPy

## Feature Extraction

* TLDExtract
* BeautifulSoup4
* Requests
* URL Parsing Utilities

## Frontend / Visualization

* Streamlit
* Plotly
* Matplotlib

## Development Tools

* Visual Studio Code
* Git & GitHub
* Joblib

---

# System Architecture

```text
Raw URL Input
        ↓
Feature Extraction Layer
        ↓
Feature Engineering & Preprocessing
        ↓
Machine Learning Models
(Random Forest / SVM / ANN / XGBoost)
        ↓
Ensemble Decision Engine
        ↓
SHAP Explainability Module
        ↓
Prediction Output
(Phishing / Legitimate)
        ↓
SOC / Dashboard Integration
```

---

# Project Structure

```text
src/
├── features/
├── training/
├── evaluation/
├── explainability/
├── api/

data/
├── raw/
├── processed/

models/
reports/
```

---

# Machine Learning Models

| Model         | Purpose                     |
| ------------- | --------------------------- |
| Random Forest | Best overall balance        |
| SVM           | Margin-based classification |
| ANN           | High recall                 |
| XGBoost       | Gradient boosting           |

---

# Feature Engineering

The phishing detection framework utilizes multiple feature categories:

## Lexical Features

* URL length
* Number of dots
* Special characters
* Digit ratio
* URL entropy

## Structural Features

* TLD analysis
* Path depth
* IP-based URLs
* Redirection behavior

## Semantic Features

* Brand impersonation
* HTTPS verification
* Suspicious webpage indicators

---

# Explainability using SHAP

PhishGuard integrates SHAP to provide:

* Feature contribution analysis
* Transparent phishing predictions
* Analyst visibility
* Explainable AI-based security insights
* SOC-friendly phishing interpretation

---

# Performance Metrics

The system is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

---

# Deployment Use Cases

* Browser-side phishing protection
* Email gateway filtering
* SIEM integration
* Security Operations Center (SOC) workflows
* Threat intelligence analysis
* Real-time phishing investigation

---

# Future Scope

* Browser extension integration
* Continual learning systems
* Cloud deployment architecture
* Mobile phishing protection
* Deep learning-based phishing detection
* Threat intelligence synchronization
* Automated phishing response workflows

---

# Conclusion

PhishGuard demonstrates a practical and explainable machine learning-based phishing detection framework capable of identifying malicious URLs using intelligent feature analysis and ensemble learning techniques.

The integration of SHAP explainability and SOC-oriented deployment concepts strengthens analyst visibility, prediction transparency, and operational cybersecurity usability.

---

# Author

**Vansh Jangra**
*Cybersecurity Research Project
*Department of Computer Science and Engineering
*Chitkara University

---

# License

This project is developed for educational, research, and defensive cybersecurity purposes only.
