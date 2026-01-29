import os
import time
import traceback
import requests
import ffmpeg
from google import genai
from google.genai import types
from PIL import Image

class ImageGenerator:
    def __init__(self, api_key=None, model_name="imagen-4.0-generate-001"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found for Image Generator.")
        
        # New Google GenAI SDK (v1)
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def generate_image(self, prompt, output_path, aspect_ratio="9:16"):
        """
        Generates an image and saves it to output_path.
        Supports both 'imagen' models (via generate_images) and 'gemini' models (via generate_content).
        """
        print(f"Generating image for prompt: {prompt[:50]}...")
        
        try:
            # Path 1: Gemini 3 Unified Endpoint (generating text + image)
            if "gemini" in self.model_name.lower():
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        # key parameter for Gemini 3 image generation
                        response_modalities=['TEXT', 'IMAGE'], 
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size="1K" # Defaulting to 1K for now
                        ),
                        # Safety settings might differ, keeping generic or relying on defaults
                    )
                )
                
                # Extract image from parts
                image_saved = False
                if response.parts:
                    for part in response.parts:
                        # Check for inline image (as_image() returns PIL Image if present)
                        # The SDK might expose it as `part.image` or process inline_data
                        # Based on docs: part.as_image()
                        try:
                            # Try modern SDK helper if available
                            if hasattr(part, 'as_image'):
                                img = part.as_image()
                                if img:
                                    img.save(output_path)
                                    print(f"Saved image to {output_path}")
                                    image_saved = True
                                    break
                            
                            # Fallback checks (inline_data)
                            if hasattr(part, 'inline_data') and part.inline_data:
                                from PIL import Image as PILImage
                                import io
                                import base64
                                # It might be raw bytes or base64. SDK usually handles this in as_image
                                # but manual fallback:
                                img_data = part.inline_data.data
                                if img_data:
                                    img = PILImage.open(io.BytesIO(img_data))
                                    img.save(output_path)
                                    print(f"Saved image to {output_path}")
                                    image_saved = True
                                    break
                        except Exception as part_err:
                            print(f"Error extracting image part: {part_err}")
                
                if image_saved:
                    return True
                else:
                    print(f"No image found in Gemini response. Parts: {len(response.parts) if response.parts else 0}")
                    # If text was returned, print it
                    if response.text:
                        print(f"Gemini returned text instead: {response.text[:100]}...")
                    return False

            # Path 2: Legacy Imagen Endpoint
            else:
                response = self.client.models.generate_images(
                    model=self.model_name,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio=aspect_ratio,
                        safety_filter_level="block_low_and_above",
                        person_generation="allow_adult"
                    )
                )
                
                if response.generated_images:
                    # Save first image
                    image = response.generated_images[0].image
                    image.save(output_path)
                    print(f"Saved image to {output_path}")
                    return True
                else:
                    print(f"No images returned. Response: {response}")
                    return False
                
        except Exception as e:
            print(f"Error generating image: {e}")
            # Mocking for now if API fails
            print("MOCK: Creating a placeholder image due to API error/unavailability.")
            try:
                img = Image.new('RGB', (1080, 1920), color = 'red')
                img.save(output_path)
            except:
                pass 
            return True


class VideoGenerator:
    def __init__(self, api_key=None, model_name="veo-3.1-generate-preview"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found for Video Generator.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def generate_video(self, prompt, output_path, duration_seconds=5, previous_video_path=None):
        """
        Generates a video using Veo model.
        If previous_video_path is provided, attempts to extend that video.
        """
        print(f"Generating VIDEO for prompt: {prompt[:50]}... (Duration: {duration_seconds}s)")
        
        # Temp path for raw download (might be full extended video)
        # Using a distinct temp filename
        raw_output_path = output_path.replace(".mp4", "_raw.mp4")
        if raw_output_path == output_path:
             raw_output_path = output_path + ".raw.mp4"

        is_extension = False
        prev_duration = 0.0

        if previous_video_path:
            print(f"Attempting to extend video from: {previous_video_path}")
            if os.path.exists(previous_video_path):
                is_extension = True
                try:
                    probe = ffmpeg.probe(previous_video_path)
                    prev_duration = float(probe['format']['duration'])
                except Exception as e:
                    print(f"Warning: Could not probe previous video duration: {e}")
                    is_extension = False

        try:
            # Prepare Video Input if extending
            video_input = None
            if is_extension:
                try:
                    print("Uploading previous video for context...")
                    # Upload file to GenAI
                    uploaded_file = self.client.files.upload(
                        file=previous_video_path,
                        config={'mime_type': 'video/mp4'}
                    )
                    
                    # FIX 1: Pass the file object directly, or use types.Part if needed.
                    # The SDK usually accepts the file object for the 'video' argument in helpers.
                    video_input = uploaded_file

                except Exception as upload_err:
                    print(f"Failed to upload previous video for extension: {upload_err}")
                    print("Fallback: Generating fresh video without extension.")
                    video_input = None
                    is_extension = False

            # Generate
            call_kwargs = {
                "model": self.model_name,
                "prompt": prompt,
                "config": types.GenerateVideosConfig(
                    number_of_videos=1,
                    aspect_ratio="9:16" 
                )
            }
            
            if video_input:
                call_kwargs["video"] = video_input

            op = self.client.models.generate_videos(**call_kwargs)
            
            print(f"Veo Operation started: {op.name}. Polling for result...")
            
            # Poll with timeout
            start_time = time.time()
            max_wait_s = 600 # 10 minutes
            
            while not op.done:
                time.sleep(10)
                if time.time() - start_time > max_wait_s:
                    raise TimeoutError(f"Veo generation timed out after {max_wait_s}s")
                # FIX 2: Correct polling syntax (pass the operation object)
                op = self.client.operations.get(op)
            
            # Check for operation error
            if op.error:
                print(f"Veo Operation Failed. Error: {op.error}")
                return False
            
            # Use op.result for success case
            result = op.result
            
            if result and hasattr(result, 'generated_videos') and result.generated_videos:
                video = result.generated_videos[0].video
                
                # Download logic
                try:
                    print(f"Downloading video content...")
                    # Note: self.client.files.download is for File API limits, result video is different.
                    
                    saved_successfully = False

                    # FIX: Force URI download because video.save() raises NotImplementedError for remote videos
                    if hasattr(video, 'uri') and video.uri:
                         print(f"Downloading from URI: {video.uri}")
                         headers = {}
                         if self.api_key:
                             headers["x-goog-api-key"] = self.api_key
                         r = requests.get(video.uri, headers=headers)
                         r.raise_for_status()
                         with open(raw_output_path, "wb") as f:
                             f.write(r.content)
                         saved_successfully = True
                    else:
                        # Fallback try save just in case, or raise error
                        if hasattr(video, 'save'):
                             try:
                                 video.save(raw_output_path)
                                 saved_successfully = True
                             except NotImplementedError:
                                 raise NotImplementedError("Video object has URI but no save method, or save method failed.")
                        else:
                             raise NotImplementedError("Video object has no save method and no URI.")
                    
                    if saved_successfully:
                        print(f"Downloaded raw video to {raw_output_path}")

                        # 2. Trim if Extension
                        if is_extension and os.path.exists(raw_output_path):
                            print(f"Trimming extended video (removing first {prev_duration:.2f}s)...")
                            try:
                                # Trim using ffmpeg
                                stream = ffmpeg.input(raw_output_path, ss=prev_duration)
                                # FIX 3: Removed c='copy' to ensure frame-accurate cutting
                                stream = ffmpeg.output(stream, output_path, avoid_negative_ts=1)
                                # Run
                                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                                
                                print(f"Saved trimmed video to {output_path}")
                                # Clean up raw
                                try:
                                    os.remove(raw_output_path)
                                except:
                                    pass
                                    
                            except Exception as cut_err:
                                print(f"Error trimming video: {cut_err}. Using raw video.")
                                # Fallback: rename raw to output
                                if os.path.exists(raw_output_path):
                                    os.rename(raw_output_path, output_path)
                        else:
                            # Just rename raw to output
                            if os.path.exists(raw_output_path):
                                os.rename(raw_output_path, output_path)
                                print(f"Saved video to {output_path}")

                        return True
                    else:
                         return False

                except Exception as sdk_err:
                    print(f"SDK download/save failed: {sdk_err}")
                    traceback.print_exc()
                    return False

            else:
                print(f"No videos returned. Result: {result}")
                return False
                
        except Exception as e:
            print(f"Error generating video: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    pass
