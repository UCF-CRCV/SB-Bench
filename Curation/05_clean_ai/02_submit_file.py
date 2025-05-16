import openai
import json
import os
from openai import AzureOpenAI
from tqdm import tqdm
import re

endpoint = os.getenv("ENDPOINT_URL", "YOUR AZURE ENDPOINT URL")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR AZURE OPENAI KEY")

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,
    api_version="2024-10-21",
)

# Upload a file with a purpose of "batch"

folder_path = 'batchFiles'
files = os.listdir(folder_path)

files = [file_name for file_name in files]
files = [os.path.join(folder_path, file) for file in files]
files.sort(key=lambda x: int(re.search(r'batch_(\d+)', x).group(1)))

file_ids = []
for file_path in tqdm(files):
  print(file_path)
  file = client.files.create(
    file=open(file_path, "rb"), 
    purpose="batch"
  )
#   print(file.model_dump_json(indent=2))
  file_id = file.id

  file_ids.append(file.model_dump_json(indent=2))

with open("file_ids.json", 'w') as f:
    json.dump(file_ids, f, indent=2)