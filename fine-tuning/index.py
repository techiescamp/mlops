import json
from sklearn.model_selection import train_test_split
from utils.token_count import num_tokens_from_message, num_assistant_tokens_from_messages, print_distribution, calculate_cost
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import time
import os

load_dotenv()

# step-1: prepare the dataset.
os.makedirs("data/raw", exist_ok=True)

with open("data/raw/dataset.jsonl", "r", encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

print("Number of examples in training dataset: ", len(dataset))

# split the datset to training and validation
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42)
print(len(train_dataset))
print(len(val_dataset))

with open("data/raw/train_dataset.jsonl", "w", encoding='utf-8') as f:
    for line in train_dataset:
        f.write(json.dumps(line) + "\n")

with open("data/raw/validate_dataset.jsonl", "w", encoding="utf-8") as f:
    for line in val_dataset:
        f.write(json.dumps(line) + "\n")

#  --- step-2: calculate token counts ---
files = ["data/raw/train_dataset.jsonl", "data/raw/validate_dataset.jsonl"]

for file in files:
    print(f"Processing file: {file}")
    with open(file, "r", encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    total_tokens = [num_tokens_from_message(ex["messages"]) for ex in dataset]
    assistant_tokens = [num_assistant_tokens_from_messages(ex["messages"]) for ex in dataset]

    print_distribution(total_tokens, "Total Tokens")
    print_distribution(assistant_tokens, "Assistant Tokens")
    print("-" * 20)

    #  cost calculation
    training_token_count, training_cost = calculate_cost(total_tokens, 8) # training cost
    input_cost = (training_token_count / 1000000) * 0.5
    output_token_count, output_cost = calculate_cost(assistant_tokens, 1.5)

    print(f"Total Tokens: {training_token_count}")
    print(f"Estimated Training Cost: ${training_cost:.2f}")
    print(f"Estimated Input Cost (inference): ${input_cost:.2f}")
    print(f"Assistant Output Tokens: {output_token_count}")
    print(f"Estimated Output Cost (inference): ${output_cost:.2f}")
    print("-" * 20)



#  --- step-3: upload fine tune files to Azure OpenAI ---
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["MODEL_API_VERSION"]
)

print(client)

#  upload the training and validation dataset files to Azure OpenAI with the SDK.
training_response = client.files.create(
    file = open("data/raw/train_dataset.jsonl", "rb"), 
    purpose = "fine-tune"
)
training_file_id = training_response.id
print(f"Training file id: {training_file_id}")
# Training file id: FileObject(id='file-0ef346fab85a4aa49ac11039db9dfca5', bytes=184530, created_at=1752258254, filename='train_dataset.jsonl', object='file', purpose='fine-tune', status='pending', expires_at=None, status_details=None)

validation_response = client.files.create(
    file = open("data/raw/validate_dataset.jsonl", "rb"),
    purpose = "fine-tune"
)
validate_file_id = validation_response.id
print(f"Validate file id: {validate_file_id}")


# --- step-4: wait for file processing ---
def wait_for_file_processing(file_id, sleep_time=2, max_wait=60):
    waited = 0
    while waited < max_wait:
        file_obj = client.files.retrieve(file_id)
        status = getattr(file_obj, "status", None)
        print(f"File {file_id} status: {status}")
        if status == "processed":
            return True
        time.sleep(sleep_time)
        waited += sleep_time
    raise TimeoutError(f"File {file_id} not processed after {max_wait} seconds.")

wait_for_file_processing(training_file_id)
wait_for_file_processing(validate_file_id)


#  --- step-5: start fine tuning model ---
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validate_file_id,
    model="gpt-35-turbo-1106",
    hyperparameters={
        "n_epochs":2,
        "seed":100,
    }
)

print(f"response: {response}")

job_id = response.id
status = response.status
print(f"JOB ID: {job_id}")
print(f"Status: {status}")
print(response.model_dump_json(indent=2))

# --- step-6: list fine-tuning jobs ---
# jobs/find_events.py

#  --- step-7: checkpoints => when each training epoch completes a checkpoint is generated. ---
# jobs/find_checkpoint.py