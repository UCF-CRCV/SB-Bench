import openai
import json
import os
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "YOUR AZURE ENDPOINT URL")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR AZURE OPENAI KEY")

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,
    api_version="2024-10-21",
)


with open('file_ids.json', "r") as file:
   data = json.load(file)

file_ids = [json.loads(d)["id"] for d in data]
file_ids.sort()

batches = []
for file_id in file_ids:

    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
    )

    batch_id = batch_response.id

    # print(batch_id)
    # print(batch_response.model_dump_json(indent=2))
    # batches.append(batch_response)

    batches.append(
        {
            "file_id": file_id,
            "batch_id": batch_id
        }
    )

    print(f"\n===============\n(File_ID: {file_id} Batch_ID: {batch_id})")

with open("batch_ids.json", "w") as file:
    json.dump(batches, file, indent=2)