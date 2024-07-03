import os
from openai import OpenAI
import streamlit as st
from PIL import Image
import io

openai = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieve image content from the file
file_id = "file-J5FMtTLoPVMBABTqAwrJeiCR"
image_response = openai.files.retrieve(file_id)
image_content = image_response["data"]

# Convert binary content to an image
image = Image.open(io.BytesIO(image_content))

# Display the image in Streamlit
st.image(image)