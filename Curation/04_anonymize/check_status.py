import os
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "YOUR AZURE ENDPOINT URL")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR AZURE OPENAI KEY")

client = AzureOpenAI(
    api_key=subscription_key,
    azure_endpoint=endpoint,
    api_version="2024-10-21",
)


batch_id = "batch_8e7ea654-f77c-4547-acc4-86cd81053af0"
found = None
for job in client.batches.list():        # ⟵ paginates under the hood
    if job.id == batch_id:
        found = job
        break

if found:
    print(f"Batch {batch_id} status: {found.status}")
    # print(found)
else:
    print(f"Batch {batch_id} not found.")

