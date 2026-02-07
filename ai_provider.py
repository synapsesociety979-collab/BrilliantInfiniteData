# ai_provider.py
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


def get_ai_response(prompt: str) -> str:
    return model.generate_content(prompt).text
