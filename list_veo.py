import os
from google import genai
import dotenv

dotenv.load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
print("Listing models...")
for m in client.models.list():
    if "veo" in m.name:
        print(f"Found: {m.name}")
