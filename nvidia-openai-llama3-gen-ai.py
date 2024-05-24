# Import necessary libraries and modules
from openai import OpenAI
from dotenv import load_dotenv
import pprint

# Load environment variables from .env file
load_dotenv()
import os

# Set OpenAI API key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

completion = client.chat.completions.create(
    model="meta/llama3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": "What can I see at NVIDIA's GPU Technology Conference?",
        }
    ],
    temperature=0.5,
    top_p=0.7,
    max_tokens=1024,
    stream=True,
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
