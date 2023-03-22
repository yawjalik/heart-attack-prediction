# Heart Attack Predictions

Model deployed [on this site](http://deploifai-app-clfjt0qgh002c512np8ymogjm-157dujxg.eastus.azurecontainer.io:7860) using [Deploifai](https://deploif.ai)

---

## Running locally

1. Install dependencies with python 3.8 and run `train.py` script to train the model. This will create an `artifacts` directory that contains the model, scaler, and encoder

2. Run main.py and visit http://localhost:7860[http://localhost:7860] to view use the model using Gradio

## Running as a Docker container

### 1. Ensure that you have created the model by running the training script

### 2. Build the image

```bash
docker build -t heartattack .
```

or use BuildKit to specify the platform for deployment

```bash
docker buildx build --platform linux/amd64 -t heartattack .
```

### 3. Run a container using the image

```bash
docker run -it -p 7860:7860 --name HeartAttack heartattack
```

### 4. Visit http://localhost:7860[http://localhost:7860]

---

## Tools used

- Python 3.8
- Pandas
- Scikit-learn
- Gradio
- Deploifai

Dataset obtained from https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
