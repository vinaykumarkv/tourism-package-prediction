# data_register.py
# Registers the raw dataset: loads it from the repo, validates the schema and prints a summary.
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Columns documented in the data dictionary (the CSV may also carry an unnamed index column)
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome",
]
TARGET = "ProdTaken"

# 1. Load the raw dataset from the repository data folder
df = pd.read_csv(RAW_PATH)

# 2. Validate that every expected column is present before the dataset counts as "registered"
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")
extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

# 3. Validate that the target only holds 0/1
if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
    raise ValueError(f"Target column '{TARGET}' must only contain 0/1 values")

# 4. Print a summary
print("Dataset registered successfully.")
print(f"Path   : {RAW_PATH}")
print(f"Rows   : {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"Extra (undocumented) columns: {extra if extra else 'none'}")
print("\nMissing values per column (only columns with gaps):")
gaps = df.isna().sum()
print(gaps[gaps > 0] if (gaps > 0).any() else "none")
print(f"\nDuplicate CustomerIDs: {df['CustomerID'].duplicated().sum()}")
print(f"\nTarget distribution ({TARGET}):")
print(df[TARGET].value_counts().rename({0: "0 (not purchased)", 1: "1 (purchased)"}))
print(f"Purchase rate: {df[TARGET].mean():.1%}")
print("\nColumn types:")
print(df.dtypes.to_string())
