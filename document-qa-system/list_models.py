import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("ERROR: Set GOOGLE_API_KEY in .env file")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

print("\nAvailable Models:")
for model in genai.list_models():
    print(f"- {model.name}")
    print(f"  Supports: {model.supported_generation_methods}\n")

# Check specific model
TARGET_MODEL = "models/gemini-1.5-pro-latest"
try:
    model = genai.get_model(TARGET_MODEL)
    print(f"\n{TARGET_MODEL} supports:")
    print(model.supported_generation_methods)
except Exception as e:
    print(f"\nError checking {TARGET_MODEL}: {str(e)}")
