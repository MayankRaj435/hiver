from pydantic import BaseModel
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

class DummyDraft(BaseModel):
    intent: str

load_dotenv('.env')
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
try:
    client.models.generate_content(
        model=os.getenv('GEMINI_MODEL'),
        contents='test',
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=DummyDraft
        )
    )
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
