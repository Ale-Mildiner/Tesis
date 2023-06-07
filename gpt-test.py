#%%
import openai
import os
import requests
key = ""
openai.api_key = os.getenv(key)
#response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)

url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer '
}
data = {
    'model': 'gpt-3.5-turbo',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'I want the answer be a list of python. Sperate in charaters the flowing sentence "Hi how are you?"'}
    ]
}

# Realizar la solicitud a la API
response = requests.post(url, headers=headers, json=data)
result = response.json()

# Obtener la respuesta del modelo
response_text = result['choices'][0]['message']['content']
print(response_text)
