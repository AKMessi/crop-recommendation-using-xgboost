# 🌾 Smart Crop Recommendation System (Hybrid AI)

A Precision Agriculture engine that recommends the optimal crop and fertilizer plan based on soil chemistry and weather data. Achieved **99.86% Accuracy** using a Hybrid Architecture (XGBoost + Rule-Based Logic).

## 🚀 Key Features
- **Hybrid Intelligence:** Combines Machine Learning (for complex crop prediction) with an Agronomic Expert System (for safe fertilizer prescriptions).
- **Explainable AI (XAI):** Integrated "Interrogation Engine" that explains *why* a crop was chosen (e.g., *"Too cold for Rice, so Coffee was selected"*).
- **Digital Twin Ready:** Simulates real-time inference using satellite weather/soil APIs.

## 🛠️ Tech Stack
- **Core:** Python 3.10, XGBoost, Scikit-Learn
- **Interpretability:** Custom Feature Attribution Engine
- **Data:** 2,200 labeled agronomic samples

## 📊 Performance
| Metric | Score |
| :--- | :--- |
| Accuracy | 99.86% |
| Safety Score | 100% (Rule-Based Validator) |
| Inference Latency | <50ms |

## ⚡ Quick Start
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the digital twin simulation:
   `python src/main.py`

## 🧠 The Logic
The system uses a **Decision Tree Ensemble** to map non-linear relationships between:
- Nitrogen, Phosphorus, Potassium
- Temperature, Humidity, Rainfall, pH# crop-recommendation-using-xgboost
