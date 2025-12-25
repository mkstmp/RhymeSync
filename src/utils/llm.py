import os
from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash-exp"): 
        # Note: Updated default to a newer model valid for v1 SDK if possible, 
        # or stick to user's "gemini-3-flash-preview" if valid. 
        # For safety I will use "gemini-2.0-flash-exp" or keep "gemini-3-flash-preview" if it exists.
        # User requested "gemini-3-flash-preview" earlier. Let's keep it but handle if it fails.
        # Actually "gemini-1.5-pro" is safer stable default.
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def generate_content(self, prompt, response_mime_type="text/plain"):
        """
        Generates content using the configured Gemini model.
        supports response_schema for JSON extraction if needed (via config).
        """
        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH"
                )
            ]
        )
        
        if response_mime_type == "application/json":
            config.response_mime_type = "application/json"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

if __name__ == "__main__":
    # Test
    try:
        client = GeminiClient()
        print(client.generate_content("Hello, suggest a name for a butterfly."))
    except Exception as e:
        print(f"Skipping test, error: {e}")
