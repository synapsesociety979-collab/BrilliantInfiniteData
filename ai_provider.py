# ai_provider.py
import os
from groq import Groq

# Use GROQ_API_KEY from environment secrets
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def get_ai_response(prompt: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "rate_limit" in error_msg:
            return "ERROR: Groq API Rate Limit exceeded. Please wait a moment."
        
        return f"ERROR: AI generation failed. Details: {str(e)}"
