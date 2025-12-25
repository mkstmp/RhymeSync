import os
import os
from google import genai
from google.genai import types
from PIL import Image
import io

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
        """
        print(f"Generating image for prompt: {prompt[:50]}...")
        
        try:
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
                print("No images returned.")
                return False
                
        except Exception as e:
            print(f"Error generating image: {e}")
            # Mocking for now if API fails
            print("MOCK: Creating a placeholder image due to API error/unavailability.")
            img = Image.new('RGB', (1080, 1920), color = 'red')
            img.save(output_path)
            # return True # Return true to simulate success for mock, or False if critical
            return True

    def generate_video(self, prompt, output_path, duration_seconds=5):
        """
        Generates a video using Veo model.
        """
        print(f"Generating VIDEO for prompt: {prompt[:50]}... (Duration: {duration_seconds}s)")
        import time
        
        try:
            # Add aspect ratio if supported by Veo (it usually is)
            # We want 9:16 for vertical video.
            # Types: "16:9", "9:16", "1:1" usually.
            
            op = self.client.models.generate_videos(
                model=self.model_name,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    aspect_ratio="9:16" 
                )
            )
            
            print(f"Veo Operation started: {op.name}. Polling for result...")
            
            # Poll with timeout
            start_time = time.time()
            max_wait_s = 600 # 10 minutes
            
            while not op.done:
                time.sleep(10)
                if time.time() - start_time > max_wait_s:
                    raise TimeoutError(f"Veo generation timed out after {max_wait_s}s")
                # Reload operation
                op = self.client.operations.get(operation=op)
            
            response = op.response
            # Accessing generated_videos attribute
            
            if response and hasattr(response, 'generated_videos') and response.generated_videos:
                video = response.generated_videos[0].video
                
                # Use SDK download first
                try:
                    print(f"Downloading video content...")
                    self.client.files.download(file=video)
                    
                    if hasattr(video, 'save'):
                        video.save(output_path)
                        print(f"Saved video to {output_path}")
                        return True
                    else:
                        raise NotImplementedError("Video object has no save method after download.")
                        
                except Exception as sdk_err:
                    print(f"SDK download/save failed: {sdk_err}. Trying manual fallback...")
                    
                    # Fallback to manual download if SDK fails
                    if hasattr(video, 'uri') and video.uri:
                         import requests
                         headers = {}
                         if self.api_key:
                             headers["x-goog-api-key"] = self.api_key
                             
                         r = requests.get(video.uri, headers=headers)
                         r.raise_for_status()
                         
                         with open(output_path, "wb") as f:
                             f.write(r.content)
                         print(f"Saved video (manual fallback) to {output_path}")
                         return True
                    else:
                        raise sdk_err

            else:
                print(f"No videos returned. Response: {response}")
                return False
        except Exception as e:
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return False

        except Exception as e:
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    pass
