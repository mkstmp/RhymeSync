
import click
import yaml
import os
import json
import torch
import sys
import ffmpeg
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print(f"DEBUG: GEMINI_API_KEY present: {'GEMINI_API_KEY' in os.environ}")
# print(f"DEBUG: Env keys: {list(os.environ.keys())}")

# Monkey patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility
# This is required because WhisperX/Pyannote use legacy pickling that is now restricted.
_original_load = torch.load
def _safe_load(*args, **kwargs):
    # FORCE weights_only to False to override library defaults
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from src.audio.aligner import AudioAligner
from src.agents.director import DirectorAgent
from src.agents.visualizer import VisualizerAgent
from src.visuals.generator import ImageGenerator
from src.visuals.text_renderer import TextRenderer
from src.video.compositor import VideoCompositor
from src.utils.subtitle import generate_srt
from src.agents.marketing import MarketingAgent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.command()
@click.option('--config', 'config_path', default='config.yaml', help='Path to config file')
@click.option('--step', type=click.Choice(['all', 'align', 'segment', 'direct', 'screenwrite', 'visualize', 'render', 'compose']), default='all', help='Execute specific step')
@click.option('--run-id', default=None, help='Unique ID for this run (default: timestamp)')
@click.option('--force', is_flag=True, help='Force re-execution of steps even if artifacts exist')
# Overrides
@click.option('--audio', 'audio_override', help='Override audio_input_file')
@click.option('--lyrics', 'lyrics_override', help='Override lyrics_file')
@click.option('--subject', 'subject_override', help='Override subject prompt')
def main(config_path, step, run_id, force, audio_override, lyrics_override, subject_override):
    """
    RhymeSync CLI - Automated Music Video Generator
    """
    # Load Config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        click.echo(f"Loaded config from {config_path}")
    else:
        # Minimal fallbacks if config is missing but overrides are present
        config = {"project": {"output_dir": "output"}, "audio": {}}
        
    # --- Apply Overrides ---
    if audio_override:
        if 'audio' not in config: config['audio'] = {}
        config['audio']['audio_input_file'] = audio_override
        click.echo(f"Override: Audio = {audio_override}")
    if lyrics_override:
        if 'audio' not in config: config['audio'] = {}
        config['audio']['lyrics_file'] = lyrics_override
        click.echo(f"Override: Lyrics = {lyrics_override}")
    if subject_override:
        config['subject'] = subject_override
        click.echo(f"Override: Subject = {subject_override}")

    # Paths & Validation
    audio_file = config.get('audio', {}).get('audio_input_file')
    lyrics_file = config.get('audio', {}).get('lyrics_file')
    if not audio_file or not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}")
        return
    if not lyrics_file or not os.path.exists(lyrics_file):
        click.echo(f"Error: Lyrics file not found: {lyrics_file}")
        return

    base_output_dir = config.get("project", {}).get("output_dir", "output")
    
    # Derive poem name
    poem_name = os.path.splitext(os.path.basename(lyrics_file))[0]
    
    # Generate Run ID
    if not run_id:
        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    output_dir = os.path.join(base_output_dir, poem_name, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output Directory: {output_dir}")
    
    # Save Effective Config
    with open(os.path.join(output_dir, "run_config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    # Ensure dirs
    os.makedirs(os.path.join(output_dir, "assets", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "assets", "text"), exist_ok=True)
    
    # Persistent State Paths
    timestamps_path = os.path.join(output_dir, "timestamps.json")
    style_bible_path = os.path.join(output_dir, "style_bible.json")
    segments_path = os.path.join(output_dir, "segments.json")
    
    # --- Step 1: Align ---
    if step in ['all', 'align']:
        if not force and os.path.exists(timestamps_path):
            click.echo(f"Skipping Step 1: Align (Artifact exists: {timestamps_path})")
        else:
            click.echo("--- Step 1: Audio Alignment ---")
            aligner = AudioAligner(config)
            aligned_data = aligner.align(audio_file, lyrics_file)
            aligner.save_timestamps(aligned_data, timestamps_path)
            
            # --- Step 1-B: Refine Text ---
            if os.path.exists(lyrics_file):
                from src.agents.text_refiner import TextRefinerAgent
                click.echo("--- Step 1-B: Text Refinement ---")
                with open(lyrics_file, "r") as f:
                    lyrics_text = f.read()
                    
                refiner = TextRefinerAgent()
                refined_data = refiner.refine_timestamps(aligned_data, lyrics_text)
                
                # Overwrite timestamps with refined version
                aligner.save_timestamps(refined_data, timestamps_path)
            else:
                click.echo("No lyrics file found for refinement. Using raw ASR.")
    
    # --- Step 2: Director ---
    if step in ['all', 'direct']:
        if not force and os.path.exists(style_bible_path):
             click.echo(f"Skipping Step 2: Director (Artifact exists: {style_bible_path})")
        else:
            click.echo("--- Step 2: The Director ---")
            if not os.path.exists(lyrics_file):
                click.echo(f"Lyrics file not found: {lyrics_file}")
                return
            
            with open(lyrics_file, "r") as f:
                lyrics_text = f.read()
                
            director = DirectorAgent()
            style_bible = director.create_style_bible(lyrics_text, config.get("subject", "A music video"))
            
            with open(style_bible_path, "w") as f:
                json.dump(style_bible, f, indent=2)
            
    # --- Step 1-B: Refine Text (Already done above) ---

    # --- Step 1.5: Segmentation (Intro/Outro & Grouping) ---
    if step in ['all', 'segment']:
        if not force and os.path.exists(segments_path):
             click.echo(f"Skipping Step 1.5: Segmentation (Artifact exists: {segments_path})")
        else:
            click.echo("--- Step 1.5: Segmentation ---")
            if not os.path.exists(timestamps_path):
                click.echo("Timestamps not found. Run 'align' first.")
                return

        import ffmpeg
        try:
            probe = ffmpeg.probe(audio_file)
            audio_duration = float(probe['format']['duration'])
        except Exception as e:
            click.echo(f"Warning: Could not probe audio duration: {e}. Defaulting to last timestamp + 5s.")
            audio_duration = None

        with open(timestamps_path, "r") as f:
            timestamps = json.load(f)

        segments = []
        current_time = 0.0
        
        # 1. Intro
        if timestamps:
            first_start = timestamps[0]['start']
            if first_start > 2.0: # If >2s gap at start
                click.echo(f"Adding Intro Segment (0.0 to {first_start:.2f}s)")
                segments.append({
                    "words": [],
                    "text": "(Intro Music)",
                    "start": 0.0,
                    "end": first_start,
                    "type": "intro"
                })
        
        # 2. Group Words
        if timestamps:
            current_segment = {"words": [], "start": timestamps[0]["start"], "end": timestamps[0]["end"], "type": "lyrics"}
            
            for i, w in enumerate(timestamps):
                # Check for gap
                is_gap = (w["start"] - current_segment["end"] > 0.5)
                # Check for long segment duration (>5s)
                is_long = (w["end"] - current_segment["start"] > 5.0)
                
                if is_gap or is_long:
                    # Look ahead: is the gap HUGE? (Instrumental bridge)
                    gap_size = w["start"] - current_segment["end"]
                    
                    if gap_size > 2.0:
                         # 2a. Close current lyrics segment
                         segments.append(current_segment)
                         
                         # 2b. Add Bridge Segment
                         click.echo(f"Adding Bridge Segment ({current_segment['end']:.2f} to {w['start']:.2f}s)")
                         segments.append({
                             "words": [],
                             "text": "(Instrumental)",
                             "start": current_segment['end'],
                             "end": w["start"],
                             "type": "bridge"
                         })
                         
                         # 2c. Start new lyrics segment
                         current_segment = {"words": [w], "start": w["start"], "end": w["end"], "type": "lyrics"}
                         
                    else:
                         # Normal line break.
                         # EXTEND current segment end to next word start to avoid micro-black-gaps
                         current_segment["end"] = w["start"]
                         segments.append(current_segment)
                         current_segment = {"words": [w], "start": w["start"], "end": w["end"], "type": "lyrics"}
                else:
                    current_segment["words"].append(w)
                    current_segment["end"] = w["end"]
            
            # Append last text segment
            segments.append(current_segment)

        # 3. Outro
        if timestamps and audio_duration:
            last_end = segments[-1]["end"]
            if audio_duration - last_end > 2.0:
                 click.echo(f"Adding Outro Segment ({last_end:.2f} to {audio_duration:.2f}s)")
                 segments.append({
                     "words": [],
                     "text": "(Outro Music)",
                     "start": last_end,
                     "end": audio_duration,
                     "type": "outro"
                 })
            # Ensure final segment stretches to exact duration to avoid drift/cutoff
            elif audio_duration > last_end:
                 segments[-1]["end"] = audio_duration

        # 4. Construct Text Fields for Visualizer
        for seg in segments:
            if seg["type"] == "lyrics":
                seg["text"] = " ".join([w["word"] for w in seg["words"]])
            # Ensure no missing text field
            if "text" not in seg: seg["text"] = ""

        with open(segments_path, "w") as f:
            json.dump(segments, f, indent=2)
            
    # --- Step 2: Director ---
    if step in ['all', 'direct']:
        click.echo("--- Step 2: The Director ---")
        if not os.path.exists(lyrics_file):
            click.echo(f"Lyrics file not found: {lyrics_file}")
            return
        
        with open(lyrics_file, "r") as f:
            lyrics_text = f.read()
            
        director = DirectorAgent()
        style_bible = director.create_style_bible(lyrics_text, config.get("subject", "A music video"))
        
        with open(style_bible_path, "w") as f:
            json.dump(style_bible, f, indent=2)

    # --- Step 2.5: Screenwriter (Enrich Segments) ---
    if step in ['all', 'screenwrite']:
        click.echo("--- Step 2.5: The Screenwriter ---")
        
        if not os.path.exists(segments_path):
            click.echo("Segments not found. Run 'segment' step first.")
            return
        if not os.path.exists(style_bible_path):
            click.echo("Style Bible not found. Run 'direct' step first.")
            return

        # Checkpoint check
        with open(segments_path, "r") as f:
            segments = json.load(f)
        
        # If already enriched and not forced, skip? 
        # Actually checking if "visual_description" exists in first segment is a good proxy.
        if not force and segments and "visual_description" in segments[0]:
             click.echo(f"Skipping Step 2.5: Screenwriter (Segments already enriched)")
        else:
            with open(style_bible_path, "r") as f:
                style_bible = json.load(f)
            
            from src.agents.screenwriter import ScreenwriterAgent
            screenwriter = ScreenwriterAgent()
            click.echo("Screenwriter Agent: Interpreting lyrics into visual scenes...")
            enriched_segments = screenwriter.enrich_segments(segments, style_bible)
            
            with open(segments_path, "w") as f:
                json.dump(enriched_segments, f, indent=2)
            click.echo("Segments enriched with visual descriptions.")
            
    # --- Step 3: Visualizer (Images) ---
    if step in ['all', 'visualize']:
        click.echo("--- Step 3: The Visualizer & Generator ---")
        
        if not os.path.exists(segments_path):
            click.echo("Segments not found. Run 'segment' step first.")
            return
        if not os.path.exists(style_bible_path):
            click.echo("Style Bible not found. Run 'direct' step first.")
            return

        with open(segments_path, "r") as f:
            segments = json.load(f)
        with open(style_bible_path, "r") as f:
            style_bible = json.load(f)
            
        visualizer = VisualizerAgent()
        
        # Determine if Veo is enabled
        use_veo = config.get("veo", {}).get("enabled", False)
        veo_model = config.get("veo", {}).get("model", "veo-2.0-generate-001")
        
        # Create output directory for assets
        images_dir = os.path.join(output_dir, "assets", "images")
        os.makedirs(images_dir, exist_ok=True)

        if use_veo:
            click.echo(f"Using Veo for VIDEO generation ({veo_model})")
            generator = ImageGenerator(model_name=veo_model) # ImageGenerator is a misnomer here, it handles video too
            ext = "mp4"
        else:
            click.echo(f"Using Imagen for IMAGE generation ({config.get('image_gen', {}).get('model', 'imagen-2')})")
            generator = ImageGenerator(model_name=config.get('image_gen', {}).get('model', 'imagen-2'))
            ext = "png"
            
        for i, seg in enumerate(segments):
            if seg["type"] not in ["lyrics", "intro", "outro"]:
                continue
            
            asset_name = f"scene_{i:03d}.{ext}"
            asset_path = os.path.join(images_dir, asset_name)
            
            # Store asset path in segment for compositor
            seg["asset_path"] = asset_path
            
            # Check if exists
            if not force and os.path.exists(asset_path):
                click.echo(f"Skipping Segment {i+1} (Exists)")
                continue

            click.echo(f"Processing Segment {i+1}/{len(segments)} [{seg['type']}]: {seg.get('text', '')}")
            
            # Context
            previous_context = ""
            if i > 0: 
                previous_context = segments[i-1].get("visual_description", segments[i-1].get("text", ""))

            visual_desc = seg.get("visual_description", "")
            prompt = visualizer.generate_prompt(seg['text'], style_bible, previous_context, visual_description=visual_desc)
            
            if use_veo:
                duration = seg["end"] - seg["start"]
                generator.generate_video(prompt, asset_path, duration_seconds=duration)
            else:
                generator.generate_image(prompt, asset_path)
                
        # Save updated segments with asset paths
        with open(segments_path, "w") as f:
            json.dump(segments, f, indent=2)

    # --- Step 4: Text Rendering ---
    if step in ['all', 'render']:
        click.echo("--- Step 4: Text Rendering ---")
        if not os.path.exists(segments_path):
             click.echo("Segments not found. Run 'visualize' step first.")
             return
             
        with open(segments_path, "r") as f:
            segments = json.load(f)
            
        renderer = TextRenderer(config)
        
        for i, seg in enumerate(segments):
            # Only render text for actual lyrics, skip Intro/Outro/Bridge labels
            if seg.get("type", "lyrics") != "lyrics":
                continue

            txt_filename = f"text_{i:03d}.png"
            txt_path = os.path.join(output_dir, "assets", "text", txt_filename)
             
            renderer.render_text_overlay(seg["text"], txt_path)
            seg["text_img"] = txt_path
        # Save updated segments with image paths
        with open(segments_path, "w") as f:
            json.dump(segments, f, indent=2)

    # --- Step 5: Compose ---
    if step in ['all', 'compose']:
        click.echo("--- Step 5: Composition ---")
        if not os.path.exists(segments_path):
             click.echo("Segments not found.")
             return
        with open(segments_path, "r") as f:
            segments = json.load(f)
            
        compositor = VideoCompositor(config)        # Output path
        final_output_path = os.path.join(output_dir, f"{poem_name}.mp4")
        
        compositor.create_video(segments, audio_file, final_output_path)
        click.echo(f"Video Render Complete: {final_output_path}")
        
        # Generate Subtitles
        srt_content = generate_srt(segments)
        srt_path = os.path.join(output_dir, f"{poem_name}.srt")
        with open(srt_path, "w") as f:
            f.write(srt_content)
        click.echo(f"Subtitles Generated: {srt_path}")
        
        # Generate YouTube Metadata
        click.echo("Generating YouTube Metadata...")
        try:
            marketing_agent = MarketingAgent()
            
            # Read lyrics
            raw_lyrics = ""
            if os.path.exists(lyrics_file):
                with open(lyrics_file, "r") as f:
                    raw_lyrics = f.read()
            
            meta_content = marketing_agent.generate_metadata(raw_lyrics, config.get("subject", ""), poem_name)
            
            if meta_content:
                meta_path = os.path.join(output_dir, f"{poem_name}_metadata.txt")
                with open(meta_path, "w") as f:
                    f.write(meta_content)
                click.echo(f"Metadata Generated: {meta_path}")
        except Exception as e:
            click.echo(f"Warning: Metadata generation failed: {e}")
            
    click.echo("Done!")

if __name__ == "__main__":
    main()
