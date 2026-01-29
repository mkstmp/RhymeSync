import re

def normalize_text(text):
    """
    Normalizes text for comparison: lowercased, removed punctuation.
    """
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text.split()

def align_words_to_lines(words, lines):
    """
    Aligns a flat list of timestamped words to a list of text lines.
    
    Args:
        words (list): List of dicts [{'word': '...', 'start': 0.0, 'end': 0.5}, ...]
        lines (list): List of strings ["First line text", "Second line text"]
        
    Returns:
        list: List of segment dicts [{'text': '...', 'start': 0.0, 'end': 0.0, 'words': [...]}]
    """
    segments = []
    
    # We will consume words from the start of the list
    word_idx = 0
    total_words = len(words)
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean: 
            continue
            
        target_tokens = normalize_text(line_clean)
        if not target_tokens:
            continue
            
        segment_words = []
        
        # Greedily match words to this line
        # We try to match the sequence of tokens. 
        # Since ASR might be imperfect, this is a best-effort greedy match.
        
        matched_count = 0
        
        # Look ahead to find where this line ends in the word list
        # We'll just take words until we satisfy the line or run out.
        # Ideally we'd use Needleman-Wunsch or something but let's stick to greedy for now
        # as the user is "suggesting" lines that should match the audio.
        
        matched_tokens_so_far = 0
        
        start_word_idx = word_idx
        
        while word_idx < total_words and matched_tokens_so_far < len(target_tokens):
            w = words[word_idx]
            w_token = normalize_text(w['word'])
            
            # If the word resulted in multiple tokens (rare but possible), or 0 tokens (punctuation)
            if not w_token:
                # Just include it in the current segment if it's punctuation attached to previous check? 
                # Or just skip. Let's include it.
                segment_words.append(w)
                word_idx += 1
                continue
            
            # Simple check: does this word likely match the next expected token?
            # Or are we just consuming words? 
            # STRICT MODE: We assume the user text roughly matches the transcript text order.
            
            segment_words.append(w)
            # We count progress by words consumed? 
            # Or just blindly assume the user wants to break roughly here.
            # Actually, the user might provide text that DOESNT match the transcript perfectly (typos).
            # So rely on Count. 
            
            matched_tokens_so_far += len(w_token)
            word_idx += 1
        
        # If we didn't find any words (maybe end of audio?), but line exists
        if not segment_words:
            # Create a placeholder or skip?
            # Let's skip empty segments to avoid crashes
            continue
            
        # Create segment
        start_time = segment_words[0]['start']
        end_time = segment_words[-1]['end']
        
        segments.append({
            "text": line_clean,
            "start": start_time,
            "end": end_time,
            "words": segment_words,
            "type": "lyrics"
        })
        
    # If there are leftover words, should we append them? 
    # Maybe add them to the last segment or create a new "overflow" segment.
    # For now, let's ignore them or users can fix their text file.
    if word_idx < total_words:
        print(f"Warning: {total_words - word_idx} words were left over after matching suggested segments.")
        print(f"Leftover words: {[w['word'] for w in words[word_idx:]]}")
    
    return segments
