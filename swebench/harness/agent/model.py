from openai import OpenAI

class AgentProxy:
    def __init__(self, name: str, base_url: str = None, 
                 api_key: str = None, temperature: float = 0.0, 
                 max_tokens: int = None, top_p: float = 1.0):
        """
        Initialize agent proxy.

        Args:
            name (str): agent type
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.model = name
        self.score = 0

    def generate(self, prompt: list[dict], offline: bool = False):
        if offline:
            raise NotImplementedError("Offline server is not implemented yet.")
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            response = response.choices[0].message.content
        return response
