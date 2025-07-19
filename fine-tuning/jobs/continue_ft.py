import json
from sklearn.model_selection import train_test_split
from token_count import num_tokens_from_message, num_assistant_tokens_from_messages, print_distribution, calculate_cost


with open("updated_dataset.jsonl", "r", encoding="utf-8") as f:
    updated_dataset = []
    for (i, line) in enumerate(f):
        try:
            data = json.loads(line)
            updated_dataset.append(data)
        except json.JSONDecodeError as e:
            print(f"Error on line - {i+1}: {e}")
            print(f"Problem on this line content (first 100 chars): {line[:100]}")
            break

print("Number of examples in training dataset: ", len(updated_dataset))

#  lets split the datset to training and validation
updated_train_dataset, updated_val_dataset = train_test_split(updated_dataset, test_size=0.2, shuffle=True, random_state=42)
print(len(updated_train_dataset))
print(len(updated_val_dataset))

with open("updated_train_dataset.jsonl", "w", encoding='utf-8') as f:
    for line in updated_train_dataset:
        f.write(json.dumps(line) + "\n")

with open("updated_validate_dataset.jsonl", "w", encoding="utf-8") as f:
    for line in updated_val_dataset:
        f.write(json.dumps(line) + "\n")

#  calculate token counts
files = ["updated_train_dataset.jsonl", "updated_validate_dataset.jsonl"]

for file in files:
    print(f"Processing file: {file}")
    with open(file, "r", encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    total_tokens = []
    assistant_tokens = []

    for ex in dataset:
        messages = ex.get("messages", {})
        total_tokens.append(num_tokens_from_message(messages))
        assistant_tokens.append(num_assistant_tokens_from_messages(messages))
    print_distribution(total_tokens, "Total Tokens")
    print_distribution(assistant_tokens, "Assistant Tokens")
    print("-" * 20)

    #  cost calculation
    training_token_count, training_cost = calculate_cost(total_tokens, 5.5) # training cost
    input_cost = (training_token_count / 1000000) * 0.44
    output_token_count, output_cost = calculate_cost(assistant_tokens, 1.76)

    print(f"🔢 Total Tokens: {training_token_count}")
    print(f"💰 Estimated Training Cost: ${training_cost:.2f}")
    print(f"💰 Estimated Input Cost (inference): ${input_cost:.2f}")
    print(f"💬 Assistant Output Tokens: {output_token_count}")
    print(f"💰 Estimated Output Cost (inference): ${output_cost:.2f}")
    print("-" * 20)



#  upload fine tune files
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["MODEL_API_VERSION"]
)

print(client)

updated_training_file_name = "updated_train_dataset.jsonl"
updated_validate_file_name = "updated_validate_dataset.jsonl"

#  upload the training and validation dataset files to Azure OpenAI with the SDK.
training_response = client.files.create(
    file = open(updated_training_file_name, "rb"), 
    purpose = "fine-tune"
)
training_file_id = training_response.id
print(f"Training file id: {training_file_id}")
# Training file id: FileObject(id='file-0ef346fab85a4aa49ac11039db9dfca5', bytes=184530, created_at=1752258254, filename='train_dataset.jsonl', object='file', purpose='fine-tune', status='pending', expires_at=None, status_details=None)

validation_response = client.files.create(
    file = open(updated_validate_file_name, "rb"),
    purpose = "fine-tune"
)
validate_file_id = validation_response.id
print(f"Validate file id: {validate_file_id}")


# wait for file processing
import time

def wait_for_file_processing(client, file_id, sleep_time=2, max_wait=60):
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

# After uploading:
wait_for_file_processing(client, training_file_id)
wait_for_file_processing(client, validate_file_id)


#  start fine tuning
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validate_file_id,
    model="gpt-35-turbo-1106.ft-627d205ac0304b22ad91f86f5867520b",
    hyperparameters={
        "n_epochs": 4,
        "seed": 100,
    }
)

print(f"response: {response}")

job_id = response.id
status = response.status
print(f"JOB ID: {job_id}")
print(f"Status: {status}")
print(response.model_dump_json(indent=2))



#  track live events till fine-tune model is trained !!
import os

def wait_for_fine_tune_model_completion(client, job_id, sleep_time=20):
    event_ids = set()
    all_events = []
    os.makedirs("logs", exist_ok=True)

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"\n🆕 Job Status: {job.status}")

        # list events
        events_response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
        new_event = [e for e in events_response.data if e.id not in event_ids]

        for eve in reversed(new_event):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eve.created_at))
            print(f"⏳{timestamp} | {eve.message}")
            event_ids.add(eve.id)
            all_events.append(eve.model_dump())
        
        if job.status in ["succeeded", "failed", "cancelled"]:
            print(f"✅ Final Job completed: {job.status}")
            #  list fine-tuning events => list individual fine-tuning events that were generated during training
            with open("logs/fine_tune_events.json", "w", encoding='utf-8') as f:
                json.dump(all_events, f, indent=2)
            return job
        
        time.sleep(sleep_time)

wait_for_fine_tune_model_completion(client, job_id)

