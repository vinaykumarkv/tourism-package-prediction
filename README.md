# Wellness Tourism Package - Purchase Prediction (MLOps)

End-to-end MLOps project for **Visit with Us**: predict whether a customer will purchase the new
Wellness Tourism Package, so the sales team can prioritise the right customers.

## Repository layout

```
.github/workflows/pipeline.yml          # CI/CD pipeline (GitHub Actions)
tourism_project/
  data/tourism.csv                      # raw dataset
  model_building/
    data_register.py                    # validates + registers the dataset
    prep.py                             # cleaning + train/test split
    train.py                            # tuning, MLflow tracking, model export
    smoke_test.py                       # deployment readiness check
  deployment/
    app.py                              # Streamlit app
    requirements.txt                    # app dependencies
    tourism_package_model_v1.joblib     # trained model (committed by the pipeline)
  requirements.txt                      # pipeline dependencies
```

## Pipeline (runs on every push to `main`)

1. **Register Dataset** - validate columns and print a summary
2. **Data Preparation** - clean, drop unneeded columns, split train/test (shared as a workflow artifact)
3. **Model Training** - tune a Random Forest, log all runs to MLflow, commit the best model to the repo
4. **Deployment Check** - load the committed model and score a sample customer

The Streamlit Community Cloud app (`tourism_project/deployment/app.py`) serves the committed model.
