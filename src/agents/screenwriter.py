import json
from src.utils.llm import GeminiClient

class ScreenwriterAgent:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        self.llm = GeminiClient(model_name=model_name)

    def enrich_segments(self, segments, style_bible):
        """
        Takes a list of segments and a Style Bible.
        Returns the same list but with a 'visual_description' field added to each segment.
        """
        # Prepare context for the LLM
        lyrics_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if seg.get("type") == "intro": text = "(Intro Music - Establish the scene)"
            if seg.get("type") == "outro": text = "(Outro Music - Final shot)"
            if seg.get("type") == "bridge": text = "(Instrumental Bridge - transition)"
            
            lyrics_data.append(f"Segment {i+1}: {text}")

        lyrics_block = "\n".join(lyrics_data)
        
        prompt = f"""
        You are the **Screenwriter** for a music video. 
        Your goal is to interpret the lyrics into concrete, detailed VISUAL descriptions for an animation team.
        
        **Style Context**:
        - Character: {style_bible.get('character', 'N/A')}
        - Setting: {style_bible.get('setting', 'N/A')}
        
        **Lyrics**:
        {lyrics_block}
        
        **Instructions**:
        1. For EACH segment, write a `visual_description`.
        2. **Interpret the Meaning**: Don't just repeat the lyrics. 
           - Example: "Bahar nikalo to mar jayegi" (Take it out, it will die) -> "The fish looks terrified/gasping for air as it is lifted out of water, or show a sad hypothetical scene."
           - Example: "Hath lagao to dar jayegi" (Touch it, it will get scared) -> "The fish shyly swims away or hides behind a coral as a hand approaches."
        3. **Keep it Consistent**: Ensure the character and setting match the Style Bible.
        4. **Output Format**: Return a JSON Object with a key "descriptions" which is a list of strings.
           - The list order MUST match the segments exactly.
           
        Example Output JSON:
        {{
          "descriptions": [
             "A wide establishing shot of the underwater reef...",
             "The orange fish swimming happily...",
             "The fish looks scared..."
          ]
        }}
        """
        
        response_text = self.llm.generate_content(prompt, response_mime_type="application/json")
        
        try:
            data = json.loads(response_text)
            descriptions = data.get("descriptions", [])
            
            if len(descriptions) != len(segments):
                print(f"Warning: generated {len(descriptions)} descriptions for {len(segments)} segments. Aligning as best as possible.")
            
            # Merit descriptors into segments
            for i, seg in enumerate(segments):
                if i < len(descriptions):
                    seg["visual_description"] = descriptions[i]
                else:
                    seg["visual_description"] = f"Visual for: {seg.get('text', '')}"
            
            return segments
            
        except Exception as e:
            print(f"Error parsing Screenwriter output: {e}")
            print(f"Raw response: {response_text}")
            return segments
