import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()      

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["MODEL_API_VERSION"]
)

response = client.chat.completions.create(
    model="gpt-35-turbo-1106-ft-8dd26f0d427349919bc40cc49fc490a7",  # This is the deployment name
    messages=[
        {"role": "system", "content": "You are a witty system design assistant. If the question isn't about system design, reply with a funny 'I don't know.'"},
        {"role": "user", "content": "What is good book to read?"}
    ],
    temperature=0.8,
    max_tokens=2000
)

print(response.choices[0].message.content)