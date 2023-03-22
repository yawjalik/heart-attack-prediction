import os
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

# Import data
data = pd.read_csv('heart-attack-dataset/heart.csv')
o2 = pd.read_csv('heart-attack-dataset/o2Saturation.csv')

# Rename 02 column and add to data
o2 = o2.rename(columns={"98.6": "o2Saturation"})
data['o2Saturation'] = o2

# Define continuous and categorical columns
cont_columns = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'o2Saturation']
cat_columns = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first')
encoded = encoder.fit_transform(data[cat_columns]).toarray()
encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_columns))
data = pd.concat([data, encoded], axis=1)

# Scale continuous columns
scaler = RobustScaler()
data[cont_columns] = scaler.fit_transform(data[cont_columns])

# Split data into X and y
X = data.drop(['output'], axis=1)
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# SVM
model = SVC()
parameters = {
    'C': [0.5, 1.0, 3.0, 10.0, 15.0, 30.0, 50.0],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    'degree': [1, 2, 3, 4]
}

searcher = GridSearchCV(model, parameters)
searcher.fit(X_train, y_train)

print(f"Best params = {searcher.best_params_}")
print(f"Best score = {searcher.best_score_}")

preds = searcher.predict(X_test)
print(f'Test accuracy: {accuracy_score(preds, y_test)}')

# Train with best params
best_model = SVC(**searcher.best_params_)
best_model.fit(X, y)

# Validate best model
preds = best_model.predict(X_test)
print(f'Test accuracy: {accuracy_score(preds, y_test)}')

# Save model, scaler, and encoder
dirname = os.path.dirname(os.path.realpath(__file__))
# Create artifacts directory if it doesn't exist
if not os.path.exists(os.path.join(dirname, "artifacts")):
    os.makedirs(os.path.join(dirname, "artifacts"))
dump(best_model, os.path.join(dirname, "artifacts", "model"))
dump(scaler, os.path.join(dirname, "artifacts", "scaler"))
dump(encoder, os.path.join(dirname, "artifacts", "encoder"))