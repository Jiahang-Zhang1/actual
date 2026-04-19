import requests
import random
import time

descriptions = [
    "McDonald's #1234",
    "Uber Ride to Airport",
    "Amazon Purchase",
    "Netflix Subscription",
    "Shell Gas Station",
    "Walmart Grocery",
    "Starbucks Coffee",
    "Delta Airlines"
]
countries = ['US', 'UK', 'AUSTRALIA']
currencies = ['USD', 'GBP', 'AUD']

SERVING_URL = "http://localhost:8080/predict"

for i in range(10):
    payload = {
        "transaction_description": random.choice(descriptions),
        "country": random.choice(countries),
        "currency": random.choice(currencies)
    }
    print(f"Request {i+1}: sending -> {payload}")
    try:
        response = requests.post(SERVING_URL, json=payload, timeout=2)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Endpoint not yet available (serving not integrated): {e}")
    time.sleep(1)
