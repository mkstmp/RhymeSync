import ffmpeg
import os

class VideoCompositor:
    def __init__(self, config):
        self.config = config
        self.resolution = tuple(config.get("video", {}).get("resolution", (1080, 1920)))
        self.fps = config.get("video", {}).get("fps", 30)

    def create_video(self, segments, audio_path, output_path):
        """
        Combines images and text based on segments using strict Concat Demuxer.
        1. Renders each segment as a temp .mp4 clip (Image + Zoom + Text).
        2. Creates a concat list file.
        3. Muxes with original audio.
        """
        import subprocess
        
        # Ensure clips dir
        clips_dir = os.path.join(os.path.dirname(output_path), "assets", "clips")
        os.makedirs(clips_dir, exist_ok=True)
        
        clip_files = []
        
        print(f"Rendering {len(segments)} intermediate clips...")
        # Prepare concat inputs
        output_dir = os.path.dirname(output_path) # Define output_dir for new logic
        
        for i, seg in enumerate(segments):
            if seg["type"] not in ["lyrics", "intro", "outro"]:
                continue
                
            duration = seg["end"] - seg["start"]
            if duration <= 0: continue # Keep original check
            
            # Asset Path: check segment first, then fallback to png/mp4 check
            asset_path = seg.get("asset_path")
            if not asset_path:
                # Fallback check
                png_path = os.path.join(output_dir, "assets", "images", f"scene_{i:03d}.png")
                mp4_path = os.path.join(output_dir, "assets", "images", f"scene_{i:03d}.mp4")
                if os.path.exists(mp4_path):
                    asset_path = mp4_path
                else:
                    asset_path = png_path
            
            if not os.path.exists(asset_path):
                print(f"Warning: Asset not found for segment {i}: {asset_path}")
                continue

            # Check if video or image
            is_video = asset_path.endswith(".mp4")
            
            # Text Overlay if exists
            # (Text renderer makes scene_XXX.png in assets/text/)
            text_path = os.path.join(output_dir, "assets", "text", f"scene_{i:03d}.png")
            has_text = os.path.exists(text_path)
            
            # Create Clip with FFmpeg (Ken Burns for IMG, Scale/Trim for VIDEO)
            clip_name = f"clip_{i:03d}.mp4"
            clip_full_path = os.path.join(clips_dir, clip_name)
            
            # Skip if exists (cache mechanism) - useful if re-running
            # if os.path.exists(clip_full_path):
            #     clip_files.append(clip_full_path)
            #     continue
            
            try:
                if is_video:
                    # Video Input
                    # Loop 5 times to be safe for duration > generated video duration
                    vid = ffmpeg.input(asset_path, stream_loop=5)
                    # Scale and Crop to Fill 1080x1920 (9:16)
                    # scale=-1:1920 ensures height fits, w adapts. or 1080:-1.
                    # force_original_aspect_ratio=increase ensures it fill box.
                    vid = vid.filter('scale', 1080, 1920, force_original_aspect_ratio="increase")
                    vid = vid.filter('crop', 1080, 1920)
                    
                    # Trim to exact segment duration
                    base_stream = vid.trim(duration=duration).setpts('PTS-STARTPTS')
                    
                else:
                    # Image Input (Ken Burns)
                    frames = int(duration * self.fps)
                    input_node = ffmpeg.input(asset_path, loop=1, t=duration)
                    
                    # Zoom Effect (Slow Zoom)
                    base_stream = input_node.filter(
                        'zoompan', 
                        z='min(zoom+0.0015,1.5)', 
                        d=frames, 
                        x='iw/2-(iw/zoom/2)', 
                        y='ih/2-(ih/zoom/2)', 
                        s=f"{self.resolution[0]}x{self.resolution[1]}",
                        fps=self.fps
                    )

                # Overlay Text
                if has_text:
                    txt_input = ffmpeg.input(text_path, loop=1, t=duration)
                    video_stream = ffmpeg.overlay(base_stream, txt_input, x=0, y=0)
                else:
                    video_stream = base_stream
                
                # Force FPS
                video_stream = video_stream.filter('fps', fps=self.fps, round='up')
                
                # Output
                out = ffmpeg.output(
                    video_stream, 
                    clip_full_path, 
                    vcodec='libx264', 
                    pix_fmt='yuv420p', 
                    t=duration  # Enforce exact duration
                )
                print(f"  Rendering Clip {i+1}: {clip_name} ({duration:.2f}s)")
                out.run(overwrite_output=True, quiet=True)
                clip_files.append(clip_full_path)
            except ffmpeg.Error as e:
                print(f"Error rendering clip {i}: {e.stderr.decode('utf8') if e.stderr else str(e)}")
                raise e

        # 2. PROPER CONCAT via Demuxer File (Avoids filter graph complexity limits and OOM)
        concat_list_path = os.path.join(clips_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for clip in clip_files:
                # Use basename because the list file is in the same directory as the clips.
                # This avoids path resolution issues (e.g. double relative paths).
                filename = os.path.basename(clip)
                safe_path = filename.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
        
        print("Concatenating clips...")
        
        # 3. Final Mux with Audio
        # ffmpeg -f concat -safe 0 -i list.txt -i audio.wav -c:v copy -c:a aac -map 0:v -map 1:a output.mp4
        
        input_audio = ffmpeg.input(audio_path)
        input_video = ffmpeg.input(concat_list_path, format='concat', safe=0)
        
        # We re-encode if we want to ensure everything is perfect, but copying strictly matches the clips.
        # But since we rendered clips with x264, we can stream copy (super fast!).
        # HOWEVER, sometimes it's safer to re-encode if timestamps are weird.
        # Let's try copy first (fastest). If issues, remove c:v copy.
        
        # Update: Re-encoding is safer for 'shortest' logic if concat duration differs slightly.
        # But 'shortest' works best if we re-encode.
        
        output = ffmpeg.output(
            input_video, 
            input_audio, 
            output_path, 
            vcodec='copy', # Stream copy video only
            acodec='aac',  # Encode audio to aac
            shortest=None
        )
        
        try:
            output.run(overwrite_output=True, quiet=False)
            print("Video Render Complete.")
        except ffmpeg.Error as e:
            print("FFmpeg Error (Concat):", e.stderr.decode('utf8') if e.stderr else str(e))
            raise e

if __name__ == "__main__":
    pass
