# Import necessary packages
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr
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


# Function to generate a response from the model
def generate_response(prompt_txt):
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
    generated_response = model.chat(messages=messages)
    generated_text = generated_response['choices'][0]['message']['content']

    return generated_text

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch()