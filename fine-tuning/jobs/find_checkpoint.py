import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["MODEL_API_VERSION"]
)

response = client.fine_tuning.jobs.list()

# Show all jobs
for job in response.data:
    print(f"Job ID: {job.id}, Status: {job.status}, Model: {job.model}")

job_id = "ftjob-627d205ac0304b22ad91f86f5867520b"

#  checkpoints => when each training epoch completes a checkpoint is generated.
checkpoint_response = client.fine_tuning.jobs.checkpoints.list(job_id)

os.makedirs("../logs/checkpoints", exist_ok=True)

with open(f"../logs/checkpoints/ft_cp_{job_id}.json", "w") as f:
    data = [cp.model_dump() for cp in checkpoint_response.data]
    json.dump(data, f, indent=2)

