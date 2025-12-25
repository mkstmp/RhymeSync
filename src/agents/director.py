import json
from src.utils.llm import GeminiClient

class DirectorAgent:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.llm = GeminiClient(model_name=model_name)

    def create_style_bible(self, lyrics_text, style_preference):
        """
        Analyzes the poem/lyrics and generates a Style Bible.
        """
        prompt = f"""
        You are the creative director for a high-end animated music video.
        
        Input:
        - Lyrics: "{lyrics_text[:1000]}" (truncated if too long...)
        - Style Preference: "{style_preference}"
        
        Your Job:
        1. Analyze the lyrics to identify the Main Character (Subject).
        2. Define a consistent Setting/Background.
        3. Create a 'Style Bible' - a precise string of visual descriptors to be used in every image prompt to ensure consistency.
        
        Output JSON format:
        {{
            "character": "Detailed visual description of the main character (approx 30 words)",
            "setting": "Detailed description of the main background/setting (approx 20 words)",
            "style_bible_suffix": "Global style keywords to append to every prompt (e.g., 'Pixar style, 3d render, octane render, soft lighting')"
        }}
        """
        
        print("Director Agent: Analyzing lyrics and creating Style Bible...")
        response_text = self.llm.generate_content(prompt, response_mime_type="application/json")
        
        try:
            # Clean up potential markdown code blocks if the model adds them (though response_mime_type should help)
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            data = json.loads(response_text)
            print("Director Agent: Style Bible Created.")
            return data
        except Exception as e:
            print(f"Director Agent Error: {e}")
            print(f"Raw Response: {response_text}")
            # Fallback
            return {
                "character": "A cute character",
                "setting": "A colorful background",
                "style_bible_suffix": style_preference
            }
