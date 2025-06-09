from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# ========== Metrik: Monitoring API Inference ==========
REQUEST_COUNT = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'inference_request_duration_seconds',
    'Time spent processing inference request'
)

REQUEST_SIZE = Histogram(
    'inference_request_size_bytes',
    'Size of inference request payload in bytes'
)

RESPONSE_SIZE = Histogram(
    'inference_response_size_bytes',
    'Size of inference response payload in bytes'
)

# ========== Metrik: Monitoring Sistem ==========
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Current CPU usage percentage')
RAM_USAGE = Gauge('system_ram_usage_percent', 'Current RAM usage percentage')

# ========== Endpoint Prometheus ==========
@app.route('/metrics')
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ========== Endpoint Inference Proxy ==========
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.get_data() 
    json_data = request.get_json()

    request_size = len(data)
    REQUEST_SIZE.observe(request_size) 

    try:
        response = requests.post("http://127.0.0.1:5000/invocations", json=json_data)
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)

        # Ukur ukuran response
        response_size = len(response.content)
        RESPONSE_SIZE.observe(response_size)

        # Catat request sukses
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.path,
            http_status=response.status_code
        ).inc()

        return jsonify(response.json()), response.status_code

    except Exception as e:
        # Catat error
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.path,
            http_status=500
        ).inc()

        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)