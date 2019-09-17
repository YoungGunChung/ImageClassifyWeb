import requests
from flask import Flask, request, jsonify
import json

jsonData = {
    "cylinders": 8,
    "displacement": 300,
    "horsepower": 78,
    "weight": 3500,
    "acceleration": 20,
    "year": 76,
    "origin": 1
}

r = requests.post("http://localhost:5000/api/mpg", json=jsonData)

if r.status_code == 200:
    print("Success: {}".format(r.text))
else:
    print("Failure: {}".format(r.text))

rJson = json.loads(r.text)
print(rJson["mpg"])
