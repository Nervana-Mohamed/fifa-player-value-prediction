# FIFA Players: Value Prediction and Rating Tiers

Machine learning notebook on the FIFA players dataset (`Fifa.csv`, 19,667 players, 9 columns).

## What's inside

- EDA: missing values, target distribution, correlations, rating by position
- Preprocessing: log transform of the target, IQR capping, one-hot encoding, scaling
- Regression: polynomial regression (degrees 1-4) with Ridge and Lasso to predict `Value Per M$`
- Classification: players grouped into 4 performance tiers from `Overall_Rating`, using Logistic Regression and Naive Bayes (Gaussian, Bernoulli, Complement)
- K-Fold and Stratified K-Fold cross-validation, plus a short discussion of the results

## Run it

```
pip install -r requirements.txt
jupyter notebook fifa_ml_analysis.ipynb
```

Keep `Fifa.csv` in the same folder as the notebook.
