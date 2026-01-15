import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report

# load
print("--- Loading Production Model ---")
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("model_artifacts/crop_recommender.json")
loaded_le = joblib.load("model_artifacts/label_encoder.pkl")


df = pd.read_csv('data/main_data.csv')
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_true = df['label']

# predict
print(f"--- Auditing all {len(df)} samples ---")

preds_idx = loaded_model.predict(X)
preds_labels = loaded_le.inverse_transform(preds_idx)

# final scorecard
full_acc = accuracy_score(y_true, preds_labels)
print(f"✅ Full Dataset Accuracy: {full_acc * 100:.2f}%")

# identifying the mistakes
errors = df[y_true != preds_labels]
if not errors.empty:
    print(f"\n⚠️ Found {len(errors)} Misclassifications:")
    # Create a clean view of errors
    error_view = errors[['label']].copy()
    error_view['Predicted'] = preds_labels[y_true != preds_labels]
    print(error_view)
else:
    print("\n🎉 Perfect Score! No errors found.")