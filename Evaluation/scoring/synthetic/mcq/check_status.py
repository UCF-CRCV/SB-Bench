import os
import json
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "YOUR AZURE ENDPOINT URL")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR AZURE OPENAI KEY")

client = AzureOpenAI(
    api_key=subscription_key,
    azure_endpoint=endpoint,
    api_version="2024-10-21",
)

with open('batch_ids.json', "r") as file:
   data = json.load(file)

batch_ids = [d["batch_id"] for d in data]
batch_ids.sort()

found = 0
for job in client.batches.list():        # ⟵ paginates under the hood
    if job.id in batch_ids:
        print(f"Batch {job.id} status: {job.status}")
        found += 1
        if found == len(batch_ids):
            break


# batch_id = "batch_2727447f-a73d-47e8-84ee-8c36eb383bcd"
# found = None
# for job in client.batches.list():        # ⟵ paginates under the hood
#     if job.id == batch_id:
#         found = job
#         break

# if found:
#     print(f"Batch {batch_id} status: {found.status}")
#     # print(found)
# else:
#     print(f"Batch {batch_id} not found.")

