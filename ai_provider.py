# ai_provider.py
import os
import google.generativeai as genai

# Use GEMINI_API_KEY from environment secrets
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Using gemini-2.0-flash-exp or gemini-1.5-flash
# Let's use gemini-1.5-flash which is widely available
model = genai.GenerativeModel("gemini-1.5-flash")

def get_ai_response(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        else:
            # Handle cases where response might be blocked or empty
            return "ERROR: AI response was blocked or empty."
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg:
            return "ERROR: AI Quota exceeded. Please wait a minute or upgrade your Gemini API plan."
        elif "404" in error_msg:
            return f"ERROR: AI Model not found. Details: {str(e)}"
        return f"ERROR: AI generation failed. Details: {str(e)}"
