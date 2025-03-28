import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from ydata_profiling.profile_report import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTEN
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("stroke_classification.csv")
data = data.drop(["pat_id"], axis=1)
target = "stroke"
x = data.drop([target], axis=1)
y = data[target]
# profile = ProfileReport(data, title="stroke report", explorative=True)
# profile.to_file("stroke_report.html")
# data['bmi'] = data['bmi'].fillna(np.median(data['bmi']))
numer = Pipeline(steps=[
    ("impute", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

cate = Pipeline(steps=[
    ("impute", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
numerical_features = ['hypertension', 'heart_disease', 'work_related_stress', 'urban_residence', 'avg_glucose_level',
                      'bmi', 'smokes']

preprocessor = ColumnTransformer(transformers=[
    ('numerical_features', numer, numerical_features),
    ('categorical_features', cate, ['gender'])
])
x = preprocessor.fit_transform(x)
ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={
    1: 800
})
x, y = ros.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))