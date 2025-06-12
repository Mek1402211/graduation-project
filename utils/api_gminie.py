from google import genai
from dotenv import load_dotenv
import os
def load_env():
    load_dotenv()
load_env()
#gemini_api_key = os.getenv("gminie_api_key")
client = genai.Client(api_key=os.getenv("gminie_api_key"))
def describe_scene2(image_path):
    file1 = client.files.upload(file=image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["give me a brife description for a blind person with the color of the objects",file1],)
    return reconsteuct_to_arabice(response.text)

def reconsteuct_to_arabice(text):
    #file1 = client.files.upload(file=text)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["reconstruct the text to arabic in more understadable way no unnecessary words or sentence.",text],)
    return response.text
