import requests
import random
import time
import os

descriptions = [
    "CHIPOTLE MEXICAN GRILL",
    "SHAKE SHACK",
    "PANERA BREAD",
    "DOMINOS PIZZA",
    "SUBWAY STORE 4521",
    "METRO NORTH RAILROAD",
    "LYFT RIDE",
    "PARKWHIZ INC",
    "TARGET STORE 0234",
    "COSTCO WHSE",
    "BEST BUY 00234",
    "VERIZON WIRELESS",
    "CONSOLIDATED EDISON",
    "SPOTIFY USA",
    "CVS PHARMACY 8823",
    "WALGREENS 6712",
    "NYC HEALTH"
]
countries = ['US', 'UK', 'AUSTRALIA']
currencies = ['USD', 'GBP', 'AUD']

SERVING_URL = os.getenv('SERVING_URL', 'http://129.114.25.80/predict')
REQUESTS_PER_SECOND = int(os.getenv('REQUESTS_PER_SECOND', '1'))
DURATION_SECONDS = int(os.getenv('DURATION_SECONDS', '60'))

print(f"Starting data generator: {REQUESTS_PER_SECOND} req/s for {DURATION_SECONDS}s")
print(f"Serving URL: {SERVING_URL}")

start_time = time.time()
i = 0

while time.time() - start_time < DURATION_SECONDS:
    payload = {
        "transaction_description": random.choice(descriptions),
        "country": random.choice(countries),
        "currency": random.choice(currencies)
    }
    print(f"Request {i+1}: sending -> {payload}")
    try:
        response = requests.post(SERVING_URL, json=payload, timeout=5)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    i += 1
    time.sleep(1.0 / REQUESTS_PER_SECOND)
