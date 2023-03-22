FROM python:3.8-slim-buster

RUN pip install --upgrade pip

COPY . .

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "main.py"]