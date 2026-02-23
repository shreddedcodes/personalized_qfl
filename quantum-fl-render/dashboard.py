from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import subprocess

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Quantum Federated Learning</title>
            <style>
                body { font-family: Arial; text-align: center; margin-top: 100px; }
                button { padding: 15px; font-size: 18px; }
            </style>
        </head>
        <body>
            <h1>Quantum Federated Learning Dashboard</h1>
            <form action="/start">
                <button type="submit">Start Federated Training</button>
            </form>
        </body>
    </html>
    """

@app.get("/start")
def start_training():
    subprocess.Popen(["python", "server.py"])
    subprocess.Popen(["python", "client_1.py"])
    subprocess.Popen(["python", "client_2.py"])

    return {"status": "Federated Training Started"}