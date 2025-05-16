import os
import json
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import openai
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


def generate_caption(path):
    df = pd.read_csv(path)
    df = df[~df['category'].isin(['Race_x_SES', 'Race_x_gender'])]
    if 'question_polarity' in df.columns:
        df = df[df['question_polarity'] == 'nonneg']

    df = df.reset_index(drop=True)

    # df = df[df['category'].isin(['Race_x_SES', 'Race_x_gender'])]
    # df = df.sample(frac=1).reset_index(drop=True)

    print(len(df['unique_question_id'].unique()))

    dl = []
    failed = []
    invalid_count = 0

    system_prompt = """
        You are an assistant tasked with anonymizing text descriptions involving sensitive personal details, including but not limited to Age, Gender Identity, Disability Status, Physical Appearance, Race/Ethnicity, Nationality, Socioeconomic Status (SES), Sexual Orientation, or Religion.

        For each provided text, your goal is to anonymize any identifiable characteristics or sensitive details, ensuring that the individuals or groups mentioned are referred to in a neutral, general way based solely on their relation to the image or scenario, without disclosing any specific personal identifiers. This means no mention of specific roles, conditions, ethnicities, or identities should remain in the text.

        **Instructions:**
        1. Replace any specific names, professions, conditions, and identifiers with image-dependent references. This can be something like "two individuals shown in the image," or "a person and their companion," without adding any additional context that could reveal identifiable characteristics.
        2. Do not include any personal identifiers, such as job roles, disabilities, or ethnicities.
        3. The output should focus only on image-dependent references—it should be clear that the text refers to the people in the image and should remain neutral and generalized.
        4. Do not add any extra context unless absolutely necessary to preserve the original meaning of the narrative.
        5. The output should maintain the narrative flow of the original text as much as possible while ensuring anonymity.
        6. Maintain a consistent narrative flow. The story should not feel "cut off" or incomplete after anonymization.
        7. anonymized_text must contain words such as people in the image, individuals in the image, or a variant of it.

        **Format:**
        The output should be in the following JSON format:
        {
            "reason": "Explanation of the change made to anonymize the text.",
            "anonymized_text": "The anonymized text that references the individuals as people in the image, individuals in the image, or a variant of it."
        }

        **Example Input/Output:**
        Example Input:
        Category: Sexual Orientation
        Text: "Even though they've each been with their significant others for several years, my pansexual friend and my gay friend had very different experiences to tell me about."

        Desired Output:
        {
            "reason": "The original text included specific references to sexual orientation. These were replaced with neutral descriptors to ensure anonymity, while maintaining the original flow and context of the story.",
            "anonymized_text": "Even though they've each been with their significant others for several years, these two individuals in the image had very different experiences to tell me about."
        }

    """

    response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "AnonymizedOutput",
                "description": "Structured response for Anonymized Text.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "anonymized_text": {"type": "string"},
                    },
                    "required": [
                        "reason",
                        "anonymized_text",
                    ],
                    "additionalProperties": False
                }
            }
        }


    batch = []
    for count, row in tqdm(df.iterrows()):
        uid, category, context, additional_metadata = row["uid"], row["category"], row["context"], row["additional_metadata_2"]
        unique_question_id = row['unique_question_id']

        if category in ['Nationality', 'Gender_identity', 'Race_etnicity', 'Race_x_gender', 'Race_x_SES']:
            prompt_eval = '\nCategory: ' + category + '\nContext: ' + context + '\nAdditional Metadata: ' + additional_metadata
        else:
            prompt_eval = '\nCategory: ' + category + '\nContext: ' + context
        
        # prompt_eval = '\nCategory: ' + category + '\nText: ' + context

        system_prompt, prompt_eval = system_prompt.strip(), prompt_eval.strip()

        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    {"type": "text","text": prompt_eval},
                ]
            },
            {"role": "system", "content": system_prompt}
        ]

        bat = {
            "custom_id": f'{unique_question_id}',
            "method": "POST", 
            "url": "/chat/completions", 
            "body": {
                    "model": deployment,
                    "messages": PROMPT_MESSAGES,
                    "max_tokens": 1024,
                    "temperature":0.7,
                    "response_format": response_format,
                },
            }
        
        batch.append(bat)

    batc_output_path = "batchFiles"
    # sub_batches = [batch[i:i + 50] for i in range(0, len(batch), 50)]
    # sub_batches = batch
    os.makedirs(batc_output_path, exist_ok=True)

    # for idx, sub_batch in enumerate(sub_batches):
    with open(f'{batc_output_path}/batch.jsonl', 'w') as f:
        for entry in batch:
            f.write(json.dumps(entry))
            f.write('\n')
    
    print(f"\n===========================\nDone making batch!")


        # To check one by one response

        # try:
        #     response = client.chat.completions.create(
        #         model = deployment,
        #         messages = PROMPT_MESSAGES,
        #         max_tokens = 1000,
        #         temperature = 0.7
        #     )

        #     t_data = {
        #         "unique_question_id": unique_question_id,
        #         "category": category,
        #         "context": context,
        #         "answer": response.choices[0].message.content,
        #     }

        #     dl.append(t_data)
        #     pprint(dl[-1])
        #     print()


        # except Exception as e:
        #     print(e)
        #     failed.append(uid)
        #     invalid_count += 1
        #     pass

        # if count == 1: 
        #     exit()




if __name__ == "__main__":
    path = "../02_scraping/AmbiguousBBQsimplifiedQid.csv"

    generate_caption(path)