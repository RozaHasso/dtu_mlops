import requests

response = requests.get('https://api.github.com/this-api-should-not-exist')
print(response.status_code)


response = requests.get('https://api.github.com')
print(response.status_code)

response = requests.get(
   'https://api.github.com/search/repositories',
   params={'q': 'requests+language:python'},
)
print(response.json())