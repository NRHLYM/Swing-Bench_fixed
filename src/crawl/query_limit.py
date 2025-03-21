import os
import requests

# Function to query rate limit for a given token
def query_rate_limit(token):
    url = "https://api.github.com/rate_limit"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to query rate limit. Status code: {response.status_code}"}

# Read tokens from a comma-separated string
tokens = os.environ["GITHUB_TOKENS"]
token_list = tokens.split(",")

# Query rate limit for each token
for token in token_list:
    result = query_rate_limit(token.strip())
    print(f"Token: {token.strip()}")
    print(result)
    print("-" * 50)