import pandas as pd, numpy as np, joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('creditcard.csv')
print(f'Dataset: {len(df)} rows | Fraud rate: {df.Class.mean()*100:.3f}%')

X = df.drop('Class', axis=1)
y = df['Class']

# Scale Amount and Time
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=1, eval_metric='auc',
    use_label_encoder=False, random_state=42)
model.fit(X_res, y_res,
          eval_set=[(X_test, y_test)], verbose=50)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Legit','Fraud']))
print(f'ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}')

joblib.dump(model, 'model/fraud_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print('Saved model/fraud_model.pkl and model/scaler.pkl')
