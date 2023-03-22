import gradio as gr
import pandas as pd
from joblib import load

cont_columns = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'o2Saturation']
cat_columns = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
scaler = load("artifacts/scaler")
encoder = load("artifacts/encoder")


def predict(X: pd.DataFrame):
    X[cont_columns] = scaler.transform(X[cont_columns])
    encoded = encoder.transform(X[cat_columns]).toarray()
    encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_columns))
    X = pd.concat([X, encoded], axis=1)

    model = load("artifacts/model")
    return model.predict(X)


def gradio_predict(age, gender, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall, o2Saturation):
    X = pd.DataFrame([[
        age,
        1 if gender == 'Male' else 0,
        0 if cp == 'TA' else 1 if cp == 'ATA' else 2 if cp == 'NAP' else 3,
        trtbps,
        chol,
        1 if fbs == 'Yes' else 0,
        0 if restecg == 'Normal' else 1 if restecg == 'ST-T wave abnormality' else 2,
        thalachh,
        1 if exng == 'Yes' else 0,
        oldpeak,
        0 if slp == 'Upsloping' else 1 if slp == 'Flat' else 2,
        caa,
        0 if thall == 'Null' else 1 if thall == 'Fixed Defect' else 2 if thall == 'Normal' else 3,
        o2Saturation,
    ]], columns=['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','o2Saturation'])

    res = predict(X)[0]

    return f'Prediction: {"Heart Failure" if res == 1 else "No Heart Failure"}'


iface = gr.Interface(
    title="Heart Failure Prediction",
    fn=gradio_predict,
    inputs=[
        gr.Slider(0, 100, 20, step=1, label="Age"),
        gr.Radio(['Male', 'Female'], value='Male', label="Gender"),
        gr.Radio(['TA', 'ATA', 'NAP', 'ASY'], value='TA', label="Chest Pain Type"),
        gr.Slider(0, 200, 120, step=1, label="Resting Blood Pressure"),
        gr.Slider(0, 600, 200, step=1, label="Serum Cholesterol"),
        gr.Radio(['Yes', 'No'], value='Yes', label="Fasting Blood Sugar > 120mg/dl"),
        gr.Radio(['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'], value='Normal', label="Resting Electrocardiographic Results"),
        gr.Slider(0, 200, 120, step=1, label="Maximum Heart Rate Achieved"),
        gr.Radio(['Yes', 'No'], value='Yes', label="Exercise Induced Angina"),
        gr.Slider(-2, 10, 0, step=0.1, label="ST Depression Induced by Exercise Relative to Rest"),
        gr.Radio(['Upsloping', 'Flat', 'Downsloping'], value='Upsloping', label="Slope of the Peak Exercise ST Segment"),
        gr.Slider(0, 3, 0, step=1, label="Number of Major Vessels (0-3) Colored by Fluoroscopy"),
        gr.Radio(['Null', 'Fixed Defect', 'Normal', 'Reversible Defect'], value='Null', label="Thalassemia"),
        gr.Slider(95, 100, 98.6, step=0.1, label="Oxygen Saturation"),
    ],
    outputs="text",
)

iface.launch(server_name="0.0.0.0")