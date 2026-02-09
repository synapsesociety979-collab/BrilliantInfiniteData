# ai_provider.py
import os
import google.generativeai as genai

# Use GEMINI_API_KEY from environment secrets
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Using gemini-2.0-flash which is widely supported and confirmed to exist
model = genai.GenerativeModel("gemini-2.0-flash")

def get_ai_response(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        else:
            return "ERROR: AI response was blocked or empty."
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg:
            return "ERROR: AI Quota exceeded. Please check your Gemini billing/quota limits at https://aistudio.google.com/app/plan_management"
        
        return f"ERROR: AI generation failed. Details: {str(e)}"
