import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

url = os.getenv('LLM_BASE_URL')
key = os.getenv('LLM_API_KEY')
model_small = os.getenv('MODEL_SMALL')

print(f'URL:   {url}')
print(f'KEY:   {key[:8]}...' if key else 'KEY: None')
print(f'MODEL: {model_small}')

client = OpenAI(base_url=url, api_key=key)

try:
    models = client.models.list()
    print('\nAVAILABLE MODELS FOR YOUR KEY:')
    for m in models.data:
        print(f' - {m.id}')
except Exception as e:
    print(f'\nERROR listing models: {e}')
