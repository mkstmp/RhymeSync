import whisperx
import json
import os
import torch

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

        print(f"Initialized AudioAligner on device: {self.device} with compute_type: {self.compute_type}")

    def align(self, audio_path, lyrics_path=None):
        """
        Transcribes and aligns audio. 
        If lyrics_path is provided, we could ideally use it for forced alignment, 
        but WhisperX's primary strength is ASR-based alignment. 
        We'll stick to WhisperX ASR + Alignment for now as it's more robust to ad-libs ("hallucinations").
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Loading audio: {audio_path}")
        audio = whisperx.load_audio(audio_path)

        # 1. Transcribe
        print("Loading Whisper model...")
        # Using medium model by default for speed/accuracy trade-off, configurable
        # Fix: Read from nested 'whisper' config
        whisper_config = self.config.get("whisper", {})
        model_size = whisper_config.get("model", "medium") if isinstance(whisper_config, dict) else "medium"
        model = whisperx.load_model(model_size, self.device, compute_type=self.compute_type)
        
        print("Transcribing...")
        result = model.transcribe(audio, batch_size=16)
        
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
    config = {"whisper_model": "tiny"}
    aligner = AudioAligner(config)
    # create a dummy file to test if not exists? No, better wait for real file.
    pass
