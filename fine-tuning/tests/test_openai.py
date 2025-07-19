from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["MODEL_API_VERSION"]
)

print(client)