import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

class StoryGenerator:
    def __init__(self, model_name="ajibawa-2023/Young-Children-Storyteller-Mistral-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
    def generate_story(self, prompt: str, max_length: int = 1000) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story

    def format_prompt(self, parameters: Dict) -> str:
        prompt = f"""Create a children's story with the following elements:
        Main Character: {parameters.get('main_character', 'a friendly dragon')}
        Setting: {parameters.get('setting', 'magical forest')}
        Theme: {parameters.get('theme', 'friendship')}
        Age Group: {parameters.get('age_group', '5-8 years')}
        
        Write a short, engaging story suitable for children:
        """
        return prompt 