# smoke_test.py
# Deployment check run by the pipeline: the committed model must load and score a sample customer.
import joblib
import pandas as pd

MODEL_PATH = "tourism_project/deployment/tourism_package_model_v1.joblib"

bundle = joblib.load(MODEL_PATH)
model, threshold, features = bundle["model"], bundle["threshold"], bundle["features"]

sample = pd.DataFrame([{
    "Age": 32, "TypeofContact": "Self Enquiry", "CityTier": 1, "DurationOfPitch": 15,
    "Occupation": "Salaried", "Gender": "Male", "NumberOfPersonVisiting": 3,
    "NumberOfFollowups": 4, "ProductPitched": "Basic", "PreferredPropertyStar": 4.0,
    "MaritalStatus": "Single", "NumberOfTrips": 3, "Passport": 1, "PitchSatisfactionScore": 4,
    "OwnCar": 1, "NumberOfChildrenVisiting": 1, "MonthlyIncome": 22000,
}])[features]

proba = float(model.predict_proba(sample)[0, 1])
assert 0.0 <= proba <= 1.0, "Probability out of range"
print(f"Smoke test passed - purchase probability {proba:.3f} "
      f"({'will purchase' if proba >= threshold else 'unlikely to purchase'}, threshold {threshold})")
