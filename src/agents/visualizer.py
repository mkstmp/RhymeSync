from src.utils.llm import GeminiClient

class VisualizerAgent:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.llm = GeminiClient(model_name=model_name)

    def generate_prompt(self, lyric_line, style_bible, previous_context=None, **kwargs):
        """
        Generates a specific image prompt for a lyric line using the Style Bible.
        """
        character_desc = style_bible.get("character", "")
        setting_desc = style_bible.get("setting", "")
        style_suffix = style_bible.get("style_bible_suffix", "")
        
        visual_desc = kwargs.get("visual_description", "")
        
        prompt = f"""
        You are the Visualizer for a music video.
        
        Context:
        - Character: {character_desc}
        - Setting: {setting_desc}
        - Current Lyric Line: "{lyric_line}"
        - Visual Action/Scene Description: "{visual_desc}"
        - Previous Scene Description: "{previous_context}"
        
        Your Job:
        Write a precise image generation prompt for Google Imagen/Flux.
        - The image MUST feature the Character in the Setting.
        - Translate the *emotion* or *action* of the lyric line into a visual scene.
        - Keep the character consistent.
        - Output ONLY the prompt string. do NOT wrap in quotes.
        """
        
        response = self.llm.generate_content(prompt)
        
        # Combine with style suffix
        full_prompt = f"{response}, {style_suffix}"
        
        # Clean up
        full_prompt = full_prompt.replace("\n", " ").strip()
        
        return full_prompt
