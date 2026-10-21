# prep.py
# Data preparation: load -> clean -> drop unnecessary columns -> train/test split -> save locally.
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# 1. Load the dataset straight from the repository data folder
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# 2. Remove unnecessary columns
#    - 'Unnamed: 0' is a leftover row index and CustomerID is a unique identifier (no predictive value)
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
#    - Designation maps 1-to-1 onto ProductPitched (e.g. Executive <-> Basic), so it adds no new
#      information. It is only dropped if that 1-to-1 relationship really holds.
if df.groupby("ProductPitched")["Designation"].nunique().max() == 1 and \
   df.groupby("Designation")["ProductPitched"].nunique().max() == 1:
    drop_cols.append("Designation")
df = df.drop(columns=drop_cols)
print("Dropped columns:", drop_cols)

# 3. Clean the data
#    a) trim stray whitespace in text columns
text_cols = df.select_dtypes(exclude="number").columns
for c in text_cols:
    df[c] = df[c].astype(str).str.strip()
#    b) fix inconsistent labels (typos in the raw file)
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})
#    c) a row without a target label cannot be used for supervised learning
df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)
#    d) missing feature values are NOT filled here: the training pipeline imputes them
#       (fitted on the training data only) so the same logic is applied at prediction time
print("\nMissing values after cleaning:", int(df.isna().sum().sum()))
print("Gender values    :", sorted(df["Gender"].unique()))
print("Occupation values:", sorted(df["Occupation"].unique()))
print("Cleaned shape:", df.shape)

# 4. Split into train and test sets (stratified so both keep the ~19% purchase rate)
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Save locally - the workflow hands these files to the next job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"\nTrain: {Xtrain.shape}  |  Test: {Xtest.shape}")
print(f"Purchase rate - train: {ytrain.mean():.3f}  test: {ytest.mean():.3f}")
print("Data prepared: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv written.")
