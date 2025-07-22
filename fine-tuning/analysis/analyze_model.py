# analyze the model
# Retrieve the file ID of the first result file from the fine-tuning job for the customized model.
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["MODEL_API_VERSION"]
)
job_id = "ftjob-627d205ac0304b22ad91f86f5867520b"

analyze_response = client.fine_tuning.jobs.retrieve(job_id)
if analyze_response.status == 'succeeded':
    result_file_id = analyze_response.result_files[0]

    retrieve = client.files.retrieve(result_file_id)

    # Download the result file.
    print(f'Downloading result file: {result_file_id}')

    with open(retrieve.filename, "wb") as file:
        result = client.files.content(result_file_id).read()
        file.write(result)
else:
    print("Fine-tuning job not succeeded yet or no result files available.")
