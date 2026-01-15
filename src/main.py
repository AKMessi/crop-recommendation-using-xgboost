import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# crop predictor
print("initializing engine 1 - ML crop predictor")
df = pd.read_csv('data/main_data.csv')

# preprocessing
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# XGBoost training
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Engine 1 Status: Active | Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

# safe SHAP plotting
try:
    print("\nGenerating Explainability Report...")
    explainer = shap.TreeExplainer(model)
    
    X_slice = X_test.iloc[:100]
    shap_values = explainer.shap_values(X_slice)
    
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_slice, plot_type="bar", show=False)
    plt.title("Key Factors Driving Crop Selection")
    plt.tight_layout()
    plt.savefig('shap_summary_robust.png')
    print(">> SHAP Summary Plot saved.")
except Exception as e:
    print(f">> SHAP Visualization skipped: {e}")

# the rule based fertilizer recommendation
print("\n--- Initializing Engine 2 (Rule-Based Logic) ---")


def recommend_fertilizer(soil_state, crop_name):
    """
    Input: soil_state (dict with N, P, K values), crop_name (str)
    Output: Recommendation String
    """
    n = soil_state['N']
    p = soil_state['P']
    k = soil_state['K']
    
    recommendations = []
    
    # logic 1: nitrogen gap (leaf growth)
    if n < 80:
        recommendations.append("Urea (High N) for leaf growth.")
    elif n > 120:
        recommendations.append("Avoid Nitrogen fertilizers to prevent burning.")

    # logic 2: phosphorus gap (roots)
    if p < 40:
        recommendations.append("DAP (Diammonium Phosphate) for root strength.")
    
    # logic 3: potassium Gap (fruiting/immunity)
    if k < 40:
        recommendations.append("MOP (Muriate of Potash) for disease resistance.")

    # logic 4: crop specifics
    if crop_name in ['rice', 'maize', 'sugarcane']:
        if n < 100: recommendations.append("Consider 14-35-14 basal application.")
    elif crop_name in ['chickpea', 'lentil', 'kidneybeans']:
        recommendations.append("Add Rhizobium culture (Bio-fertilizer) for legumes.")
    
    if not recommendations:
        return "Soil is balanced. Use generic NPK 19-19-19 maintenance dose."
        
    return " + ".join(recommendations)

# testing the hybrid pipeling
print("\nend to end pipeline")


sample_idx = 10
sample_input = X_test.iloc[sample_idx]
actual_label = le.inverse_transform([y_test[sample_idx]])[0]

# step 1: predict crop
pred_prob = model.predict_proba(sample_input.to_frame().T)
pred_idx = np.argmax(pred_prob)
predicted_crop = le.classes_[pred_idx]

print(f"Input Soil: N={sample_input['N']}, P={sample_input['P']}, K={sample_input['K']}, pH={sample_input['ph']:.1f}")
print(f"Step 1 (ML Model): Recommended Crop -> {predicted_crop.upper()} (Confidence: {np.max(pred_prob)*100:.1f}%)")

# step 2: recommend fertilizer

soil_dict = {'N': sample_input['N'], 'P': sample_input['P'], 'K': sample_input['K']}
advice = recommend_fertilizer(soil_dict, predicted_crop)

print(f"Step 2 (Rule Engine): Fertilizer Plan -> {advice}")