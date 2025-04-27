import requests

url = "http://127.0.0.1:8000/chat"
data = {"prompt": "너는 누구야?"}

response = requests.post(url, json=data)
print("Bot:", response.json()["response"])
