import requests

url = 'http://localhost:5000/predict_api'
# r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})
r = requests.post(url,json={'query' : 'Helo, how are you doin?'})
print(r.json())