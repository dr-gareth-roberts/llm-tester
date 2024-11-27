import openai
import anthropic
import os
from typing import Dict, Any
from .base_handler import BaseAPIHandler

class OpenAIHandler(BaseAPIHandler):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key

    def supports_model(self, model: str) -> bool:
        supported_models = [
            'gpt-3.5-turbo', 'gpt-4', 'text-davinci-003', 
            'text-curie-001', 'text-babbage-001'
        ]
        return model in supported_models

    def generate_text(self, 
                      prompt: str, 
                      model: str = 'gpt-3.5-turbo', 
                      **kwargs) -> str:
        try:
            # Adjust parameters based on model type
            if model.startswith('gpt'):
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    **kwargs
                )
                return response.choices[0].message.content
            else:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    **kwargs
                )
                return response.choices[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API Error: {str(e)}")

class AnthropicHandler(BaseAPIHandler):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def supports_model(self, model: str) -> bool:
        supported_models = [
            'claude-1', 'claude-2', 'claude-instant-1'
        ]
        return model in supported_models

    def generate_text(self, 
                      prompt: str, 
                      model: str = 'claude-2', 
                      **kwargs) -> str:
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                **kwargs
            )
            return response.completion
        except Exception as e:
            raise RuntimeError(f"Anthropic API Error: {str(e)}")