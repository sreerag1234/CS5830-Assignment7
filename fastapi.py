import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
import numpy as np
import argparse
from PIL import Image
from keras.models import load_model
from keras.models import Sequential
from prometheus_client import Counter, Gauge, make_asgi_app
import time

REQUEST_COUNT = Counter('request_count', 'App Request Count', ['app_name', 'client_ip'])
REQUEST_LATENCY = Gauge('request_latency', 'Request latency')
REQUEST_LATENCY_PER_CHAR = Gauge('request_latency_per_char', 'Request latency per character', ['client_ip'])
REQUEST_INPUT_LENGTH = Gauge('request_input_length', 'Request input length', ['client_ip'])
app = FastAPI()

# Prometheus ASGI app to expose metrics
metrics_app = make_asgi_app()

# Mount Prometheus ASGI app
app.mount("/metrics", metrics_app)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Saved model path")
args = parser.parse_args()
def load_model_path(path: str) -> Sequential:
    return load_model(path)
def predict_digit(model: Sequential, data_point: list) -> str:
    data_point = np.array(data_point).reshape(1, 784)
    pred = model.predict(data_point).argmax()
    return str(pred)

model = load_model_path(args.model_path)
def format_image(image):
    img_grey = image.convert('L').resize((28, 28))
    serialized_array = list(img_grey.getdata())
    return serialized_array

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host
    REQUEST_COUNT.labels(app_name='digit_recognizer', client_ip=client_ip).inc()

    start_time = time.time()
    image = Image.open(file.file)
    serialized_image = format_image(image)   
    digit = predict_digit(model, serialized_image)
    e_time = time.time()
    in_len = len(serialized_image)
    dur = e_time - start_time    
    latency_per_char = (dur / in_len) * 1e6

    REQUEST_LATENCY.set(dur)
    REQUEST_INPUT_LENGTH.labels(client_ip=client_ip).set(in_len)
    REQUEST_LATENCY_PER_CHAR.labels(client_ip=client_ip).set(latency_per_char)

    return {"digit": digit}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)