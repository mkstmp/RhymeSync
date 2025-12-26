import datetime

def format_timestamp(seconds):
    """Formats seconds into SRT timestamp format: HH:MM:SS,mmm"""
    total_seconds = int(seconds)
    millis = int((seconds - total_seconds) * 1000)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt(segments):
    """Generates SRT content from segments."""
    srt_content = ""
    index = 1
    for seg in segments:
        # Only include lyrics segments in subtitles
        if seg.get("type", "lyrics") != "lyrics":
            continue
        
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        
        if not text: 
            continue
        
        srt_content += f"{index}\n"
        srt_content += f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
        srt_content += f"{text}\n\n"
        index += 1
        
    return srt_content
