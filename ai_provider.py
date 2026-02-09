# ai_provider.py
import os
import google.generativeai as genai

# Use GEMINI_API_KEY from environment secrets
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Using gemini-1.5-flash as it has higher rate limits on paid plans
model = genai.GenerativeModel("gemini-1.5-flash")

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
            # If 1.5-flash hits quota, try a short sleep or just report it clearly
            # But the user says they have a plan, so we should ensure we aren't using an experimental model with low limits
            return "ERROR: AI Quota exceeded. Please check your Gemini billing/quota limits at https://aistudio.google.com/app/plan_management"
        
        return f"ERROR: AI generation failed. Details: {str(e)}"
