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

# Job ID: ftjob-<model-jon-ID>, Status: <succeeded | failed>, Model: <model-name>

job_id = "ftjob-627d205ac0304b22ad91f86f5867520b"

events_response = client.fine_tuning.jobs.list_events(job_id)

print(events_response.model_dump_json(indent=2))

os.makedirs("../logs/events", exist_ok=True)

with open(f"../logs/events/ft_job_{job_id}.json", "w", encoding='utf-8') as f:
    json.dump([e.model_dump() for e in events_response.data], f, indent=2)


# --- Monitor live while training ---
#  Track live events until fine-tune model is completed !! ---
#  Drawback on tracking live is that you have run the code for 1 to 2 hours approximatley.
#  You can track events aafter completing the model
#  Showd monitor of live events in two ways:

# def wait_for_fine_tune_model_completion(client, job_id, sleep_time=20):
#     event_ids = set()
#     all_events = []
#     os.makedirs("logs", exist_ok=True)

#     while True:
#         job = client.fine_tuning.jobs.retrieve(job_id)
#         print(f"\n🆕 Job Status: {job.status}")

#         # list events
#         events_response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
#         new_event = [e for e in events_response.data if e.id not in event_ids]

#         for eve in reversed(new_event):
#             timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eve.created_at))
#             print(f"⏳{timestamp} | {eve.message}")
#             event_ids.add(eve.id)
#             all_events.append(eve.model_dump())
        
#         if job.status in ["succeeded", "failed", "cancelled"]:
#             print(f"✅ Final Job completed: {job.status}")
#             #  list fine-tuning events => list individual fine-tuning events that were generated during training
#             with open("logs/fine_tune_events.json", "w", encoding='utf-8') as f:
#                 json.dump(all_events, f, indent=2)
#             return job
        
#         time.sleep(sleep_time)

# wait_for_fine_tune_model_completion(client, job_id)