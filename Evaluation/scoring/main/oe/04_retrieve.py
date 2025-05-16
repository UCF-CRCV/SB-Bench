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

with open('batch_ids.json', 'r') as file:
    data = json.load(file)

batch_ids = [d['batch_id'] for d in data]

os.makedirs('Outputs', exist_ok=True)

failed = []
for batch_id in batch_ids:
    batch_response = client.batches.retrieve(batch_id)

    output_file_id = batch_response.output_file_id

    if not output_file_id:
        output_file_id = batch_response.error_file_id

    if output_file_id:
        # loading ground truths from the input file
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')  

        final_outputs = []
        for index, raw_response in enumerate(raw_responses):
            json_response = dict(json.loads(raw_response))

            custom_id = json_response["custom_id"]

            try:
                output = json_response["response"]["body"]["choices"][0]["message"]["content"]
                # print(output)
                # print(type(output))

                # final_output = json.loads(output)
                # image_correct = final_output["image_correct"]
                # reason = final_output["reason"]
                # print(image_correct, reason)

                # print(output)
                # break
                # exit()

            except:
                print(f"unable to fetch caption for {batch_id}, {custom_id}.")
                failed.append({"custom_id": custom_id})
                continue

            final_outputs.append(
                {
                    "custom_id": custom_id,
                    "output": output
                }
            )

        print(f"\n====================\n{batch_id} fetched total samples: {len(final_outputs)}\n")

        with open(f"Outputs/{batch_id}.json", "w") as f:
            json.dump(final_outputs, f, indent=2)

        if failed:
            with open(f"failed.json", "w") as f:
                json.dump(failed, f, indent=2)
