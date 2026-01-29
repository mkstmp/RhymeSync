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
        
        # Sanitize character description for safety (remove specific ages)
        character_desc = character_desc.replace("5-year-old", "young").replace("6-year-old", "young").replace("child", "character")
        
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
        
        # NOTE: We already implemented the regex unescape logic in the previous turn?
        # Let's double check if I actually applied it or if it was checking for \\u.
        # The user says "Both issues are still present".
        # If the file content shows: \u0905\u0928... that means literal backslash u.
        # My previous regex was r'\\u([0-9a-fA-F]{4})'.
        # That matches a LITERAL \u followed by 4 hex chars.
        # If the string in python memory has ACTUAL unicode chars, this regex won't match.
        # But if the string in python memory has ESCAPE SEQUENCES (backslash u), it will match.
        
        # Let's clean this up to be universally safe.
        import re
        
        # 1. Decode literal unicode escapes (e.g. string containing "\u0906")
        def unescape_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except:
                return match.group(0)
                
        # Regex for \uXXXX
        full_prompt = re.sub(r'\\u([0-9a-fA-F]{4})', unescape_unicode, full_prompt)
        
        # Regex for \xXX (just in case)
        full_prompt = re.sub(r'\\x([0-9a-fA-F]{2})', unescape_unicode, full_prompt)

        return full_prompt
