# Import necessary packages
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from dotenv import load_dotenv
load_dotenv()
import os

# # Model and project settings
# model_id = "meta-llama/llama-3-2-11b-vision-instruct"  # Directly specifying the LLAMA3 model

# # Set credentials to use the model
# credentials = Credentials(
#                    url = "https://us-south.ml.cloud.ibm.com",
#                   )

# # Generation parameters
# params = TextChatParameters(
#     temperature=0.7,
#     max_tokens=1024
# )

# project_id = "skills-network"

# # Initialize the model
# model = ModelInference(
#     model_id=model_id,
#     credentials=credentials,
#     project_id=project_id,
#     params=params
# )

watsonx_API = os.getenv("watsonx_API") 
project_id= os.getenv("project_id") # like "0blahblah-000-9999-blah-99bla0hblah0"

params = TextChatParameters(
    temperature=0.7,
    max_tokens=1024
)

model = ModelInference(
    model_id='meta-llama/llama-3-2-11b-vision-instruct', 
    params=params,
    credentials={
        "apikey": watsonx_API,
        "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id=project_id
    )


prompt_txt = "How to be a good Data Scientist?"  # Your question

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_txt
            },
        ]
    }
]

# Attempt to generate a response using the model with overridden parameters
generated_response = model.chat(messages=messages)
generated_text = generated_response['choices'][0]['message']['content']

# Print the generated response
print(generated_text)