from PIL import Image, ImageDraw, ImageFont
import os

class TextRenderer:
    def __init__(self, config):
        self.config = config
        # self.font_path and self.font_size are now handled by _load_font
        self.resolution = tuple(config.get("video", {}).get("resolution", (1080, 1920)))

    def _load_font(self):
        """Loads a font that supports Hindi/Devanagari."""
        font_path = self.config.get("text", {}).get("font_path", None)
        font_size = self.config.get("text", {}).get("font_size", 60)
        
        # Priority 1: Configured Font
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"Error loading config font: {e}")

        # Priority 2: macOS System Font (Kohinoor - Excellent for Indic)
        # Kohinoor.ttc usually contains Devanagari. Index may vary, but 0 often works or auto-resolved.
        # Alternatively, 'Kohinoor Devanagari' might be needed if accessing by name, but path is safer.
        macos_font = "/System/Library/Fonts/Kohinoor.ttc"
        if os.path.exists(macos_font):
             try:
                # Try index 1 which is often Devanagari in the collection, or default
                # Just loading the TTC usually works for standard chars, but let's try.
                return ImageFont.truetype(macos_font, font_size)
             except Exception:
                pass
        
        macos_fallback = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
        if os.path.exists(macos_fallback):
            return ImageFont.truetype(macos_fallback, font_size)

        # Priority 3: Bundled Font (if user added it)
        bundled_font = os.path.join(os.path.dirname(__file__), '../../assets/fonts/NotoSansDevanagari.ttf')
        if os.path.exists(bundled_font):
            return ImageFont.truetype(bundled_font, font_size)
            
        # Priority 4: Linux/Generic Paths
        linux_fonts = [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "arial.ttf"
        ]
        
        for f in linux_fonts:
            if os.path.exists(f):
                return ImageFont.truetype(f, font_size)
        
        print("WARNING: No suitable Hindi font found. Text may not render correctly.")
        return ImageFont.load_default()

    def render_text_overlay(self, text, output_path):
        """
        Create a transparent PNG with the text centered or positioned.
        """
        # Create transparent image
        img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        font = self._load_font()

        # Check for Raqm support (libraqm) implicitly via Pillow
        # Pillow >= 4.2.0 uses Raqm if installed for complex scripts
        
        # Calculate text size using textbbox (newer Pillow)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position: Bottom center usually good for subtitles
        x = (self.resolution[0] - text_width) / 2
        y = self.resolution[1] - (self.resolution[1] * 0.15) # 15% from bottom

        # Draw text with shadow/outline for visibility
        shadow_color = "black"
        text_color = "white"
        outline_width = 3
        
        # Draw outline/shadow
        for adj in range(-outline_width, outline_width+1):
           for adj2 in range(-outline_width, outline_width+1):
               draw.text((x+adj, y+adj2), text, font=font, fill=shadow_color)

        draw.text((x, y), text, font=font, fill=text_color)
        
        img.save(output_path)
        print(f"Saved text overlay: {output_path}")

if __name__ == "__main__":
    # Test
    # need a valid font path to test hindi correctness visually
    pass
