import whisperx
import json
import os
import torch
from google import genai
from google.genai import types

class AudioAligner:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Force CPU if on Mac (MPS support in WhisperX/CTranslate2 is flaky/unsupported)
        if torch.backends.mps.is_available():
             print("MPS detected but WhisperX/CTranslate2 requires CPU or CUDA. Forcing CPU.")
             self.device = "cpu"
        
        # Determine compute type based on device
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.alignment_method = self.config.get("alignment", {}).get("method", "whisper")
        self.gemini_model = self.config.get("alignment", {}).get("model", "gemini-2.0-flash")

        print(f"Initialized AudioAligner on device: {self.device}")
        print(f"Alignment Method: {self.alignment_method} (Model: {self.gemini_model if self.alignment_method == 'gemini' else 'WhisperX'})")

    def align(self, audio_path, lyrics_path=None):
        """
        Transcribes and aligns audio using the configured method (WhisperX or Gemini).
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.alignment_method == "gemini":
            return self.align_with_gemini(audio_path, lyrics_path)
        else:
            return self.align_with_whisper(audio_path)

    def align_with_gemini(self, audio_path, lyrics_path):
        """
        Uses Gemini (Multimodal) to generate word-level timestamps.
        """
        print(f"Aligning with Gemini ({self.gemini_model})...")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found for Gemini Alignment.")

        client = genai.Client(api_key=api_key)
        
        # Load lyrics text if available
        lyrics_text = ""
        if lyrics_path and os.path.exists(lyrics_path):
            with open(lyrics_path, "r") as f:
                lyrics_text = f.read()
            print(f"Using provided lyrics for context.")
        else:
            print("No lyrics file provided. Asking Gemini to transcribe and timestamp.")

        # Upload audio file
        print(f"Uploading audio: {audio_path}...")
        try:
            # Check if file is small enough for direct upload or needs File API
            # For simplicity, using File API as it's robust for audio
            audio_file = client.files.upload(file=audio_path)
            print(f"Uploaded file: {audio_file.name}")
        except Exception as e:
            print(f"Error uploading audio to Gemini: {e}")
            raise

        prompt = f"""
        You are an expert audio aligner.
        
        Task: Align the provided audio with the transcript (if provided) or transcribe it with timestamps.
        Audio File: [Attached]
        Transcript Context: "{lyrics_text}"
        
        Output Format: JSON list of lists. Minified (no whitespace/newlines).
        [ ["word", start, end], ["word", start, end] ]
        
        Rules:
        - Cover the ENTIRE audio duration accurately.
        - Output ONLY the raw JSON string. NO Markdown. NO whitespace.
        """

        print("Sending request to Gemini...")
        try:
            response = client.models.generate_content(
                model=self.gemini_model,
                contents=[prompt, audio_file],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=8192
                )
            )
            
            if response.text:
                text = response.text.strip()
                if text.startswith("```json"): text = text[7:]
                if text.startswith("```"): text = text[3:]
                if text.endswith("```"): text = text[:-3]
                text = text.strip()

                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON Decode Error ({e}). Attempting repair...")
                    # Repair strategy: Find last closing bracket of an inner list ']', cut off, and close.
                    # Assuming format [[...], [...], ...]
                    last_bracket = text.rfind(']')
                    if last_bracket != -1:
                        # If string ends with something like '], ["partial...', we want the last ']' from a COMPLETE item.
                        # We look for '], [' pattern or just ']' at end of a valid item. 
                        # Using rfind(']') might find the partial one if it existed?
                        # Safer: iterate or regex.
                        # Simple heuristic: rfind(']') is risky if inside string. 
                        # But in minified JSON, ']' usually means end of list.
                        
                        # Let's try to find the last occurrence of '],[' or sequence that implies separation
                        # Actually, if we just cut at last ']', append ']', it might work if the truncation wasn't inside a list.
                        # If truncation was inside a string inside a list, last ']' is the previous item's end.
                        
                        # Fix: Trim to last ']', then add ']' to close outer list.
                        # Be careful if text[last_bracket] is the closing of the *outer* list (unlikely if error).
                        
                        # Iterate backwards to find a safe cut point.
                        # We expect pairs of brackets.
                        # Regex for valid inner list: \[ ".*?", \d+(\.\d+)?, \d+(\.\d+)? \]
                        import re
                        # Find all valid inner lists
                        matches = re.findall(r'\[\s*".*?"\s*,\s*[\d\.]+\s*,\s*[\d\.]+\s*\]', text)
                        if matches:
                            print(f"Recovered {len(matches)} valid segments from truncated JSON.")
                            # Reconstruct valid JSON
                            fixed_json = "[" + ",".join(matches) + "]"
                            data = json.loads(fixed_json)
                        else:
                            print("Could not repair JSON.")
                            raise e
                    else:
                        raise e
                    
                # Convert list-of-lists to list-of-dicts
                aligned_words = []
                for item in data:
                    if isinstance(item, list) and len(item) >= 3:
                        word = item[0]
                        start = item[1]
                        end = item[2]
                        aligned_words.append({
                            "word": str(word),
                            "start": float(start),
                            "end": float(end),
                            "score": 1.0 
                        })
                    elif isinstance(item, dict):
                        # Fallback if model ignores instruction and outputs dicts
                        word = item.get("word") or item.get("text")
                        start = item.get("start")
                        end = item.get("end")
                        if word is not None:
                            aligned_words.append({
                                "word": str(word),
                                "start": float(start),
                                "end": float(end),
                                "score": 1.0
                            })
                        
                print(f"Gemini alignment complete. {len(aligned_words)} segments found.")
                return aligned_words
                    


        except Exception as e:
            print(f"Error during Gemini Alignment: {e}")
            raise


    def align_with_whisper(self, audio_path):
        """
        Original WhisperX implementation.
        """
        print(f"Loading audio: {audio_path}")
        audio = whisperx.load_audio(audio_path)

        # 1. Transcribe
        print("Loading Whisper model...")
        whisper_config = self.config.get("whisper", {})
        model_size = whisper_config.get("model", "medium") if isinstance(whisper_config, dict) else "medium"
        model = whisperx.load_model(model_size, self.device, compute_type=self.compute_type)
        
        language = whisper_config.get("language", None)
        
        print(f"Transcribing... (Language forced: {language})" if language else "Transcribing... (Auto-detection)")
        result = model.transcribe(audio, batch_size=16, language=language)
        
        # 2. Align
        print("Loading Alignment model...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        
        print("Aligning...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        
        # 3. Process output to flat word list with timestamps
        aligned_words = []
        for segment in result["segments"]:
            for word in segment.get("words", []):
                if "start" in word and "end" in word:
                    aligned_words.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "score": word.get("score", 0)
                    })
        
        return aligned_words

    def save_timestamps(self, aligned_words, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aligned_words, f, indent=2, ensure_ascii=False)
        print(f"Saved timestamps to {output_path}")

if __name__ == "__main__":
    # Simple test
    # config = {"whisper_model": "tiny", "alignment": {"method": "gemini", "model": "gemini-2.0-flash"}}
    # aligner = AudioAligner(config)
    pass
