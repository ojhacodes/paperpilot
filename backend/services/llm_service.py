import os
from typing import List, Dict, Any, Optional
from backend.config import settings

class LLMService:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.DEFAULT_PROVIDER
        self.provider = self.provider.lower()
        
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "gemini":
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured in .env or environment.")
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.model_name = settings.GEMINI_MODEL
            
        elif self.provider == "openai":
            api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not configured in .env or environment.")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model_name = settings.OPENAI_MODEL
            
        elif self.provider == "anthropic":
            api_key = settings.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not configured in .env or environment.")
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model_name = settings.ANTHROPIC_MODEL
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Sends chat messages to the selected LLM and returns the text response.
        """
        if self.provider == "gemini":
            return self._generate_gemini(messages, system_instruction)
        elif self.provider == "openai":
            return self._generate_openai(messages, system_instruction)
        elif self.provider == "anthropic":
            return self._generate_anthropic(messages, system_instruction)
        return ""

    def _generate_gemini(self, messages: List[Dict[str, str]], system_instruction: Optional[str]) -> str:
        from google.genai import types
        
        # Convert messages to Gemini format
        # Gemini Client handles List[types.Content]
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
            
        config = types.GenerateContentConfig()
        if system_instruction:
            config.system_instruction = system_instruction
        config.temperature = 0.2
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config
        )
        return response.text

    def _generate_openai(self, messages: List[Dict[str, str]], system_instruction: Optional[str]) -> str:
        openai_messages = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction})
        
        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=0.2
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, messages: List[Dict[str, str]], system_instruction: Optional[str]) -> str:
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
            
        kwargs = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "temperature": 0.2
        }
        if system_instruction:
            kwargs["system"] = system_instruction
            
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
