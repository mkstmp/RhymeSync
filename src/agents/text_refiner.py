import json
from src.utils.llm import GeminiClient

class TextRefinerAgent:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.llm = GeminiClient(model_name=model_name)

    def refine_timestamps(self, timestamped_words, ground_truth_text):
        """
        Aligns Whisper's timestamped words with the ground truth lyrics text.
        Returns a new list of timestamped words with corrected spelling.
        """
        # Prepare input for LLM
        # We need to pass the JSON and the clean text and ask it to output corrected JSON
        # Since the list might be long, we might need to batch, but for poems it's likely fine.
        
        words_json_str = json.dumps(timestamped_words, ensure_ascii=False)
        
        prompt = f"""
        You are a "Force Aligner" and Correction expert.
        
        Input 1: A JSON list of words with timestamps (from Whisper ASR). The spelling might be wrong or phonetic.
        Input 2: The correct Ground Truth Lyrics.
        
        Task:
        1. Read the Ground Truth Lyrics.
        2. Correct the "word" fields in the JSON to match the Ground Truth spelling exactly.
        3. Maintain the "start" and "end" timestamps from the input JSON.
        4. If a word from Ground Truth is missing in the JSON, try to fit it in or merge it, but prioritize the existing timestamps.
        5. The number of words in Output JSON should ideally match the recognizable words in the Ground Truth.
        
        Ground Truth:
        "{ground_truth_text}"
        
        Input JSON:
        {words_json_str}
        
        Output:
        Return ONLY the corrected JSON list. Valid JSON only. No markdown.
        """
        
        print("TextRefiner Agent: refining timestamps with ground truth...")
        response_text = self.llm.generate_content(prompt, response_mime_type="application/json")
        
        try:
            # Clean up potential markdown
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            refined_data = json.loads(response_text)
            print(f"TextRefiner Agent: Refined {len(refined_data)} words.")
            return refined_data
        except Exception as e:
            print(f"TextRefiner Agent Error: {e}")
            print("Falling back to original timestamps.")
            return timestamped_words
