from src.utils.llm import GeminiClient
import json

class MarketingAgent:
    def __init__(self):
        self.llm = GeminiClient()

    def generate_metadata(self, lyrics, subject, poem_name):
        print("Marketing Agent: Generating YouTube Metadata...")
        
        prompt = f"""
        You are a YouTube SEO expert specializing in Kids and Nursery Rhymes content.
        Your task is to generate optimized metadata for a new music video.

        **Input Data:**
        - Poem Name: {poem_name}
        - Subject/Theme: {subject}
        - Lyrics:
        {lyrics[:2000]} # Truncate if too long, usually short

        **Required Output Format (Plain Text):**
        
        Video Title
        [Catchy Title with Emojis | Keywords | #Shorts]

        Description
        [Engaging 2-3 lines description for toddlers/parents]

        Lyrics: 
        [The Lyrics provided]

        [Call to Action, e.g. Watch more!]

        [Hashtags block]

        Tags (Copy and Paste into the "Tags" section)
        [Comma separated high-value keywords]

        **Rules:**
        - Title must be CLICKBAIT but safe for kids (Use emojis like 🦋, 🌈).
        - Targeted audience: Toddlers, Preschoolers, Indian Parents (if lyrics is Hindi).
        - Tags should be relevant (Hindi Rhymes, Balgeet, Kids Songs, etc.).
        - Output strictly the format above.
        """
        
        response = self.llm.generate_content(prompt)
        return response
