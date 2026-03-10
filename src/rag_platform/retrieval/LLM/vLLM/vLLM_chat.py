from langchain_openai import ChatOpenAI

class VLLMManager:
    def __init__(self, model: str, base_url: str = "http://localhost:13001/v1"):
        self.model = model
        self.client = ChatOpenAI(
            model=model,
            base_url=base_url,
            temperature=0,
        )

    def chat(self, messages, temperature: float = 0) -> str:
        # create a temporary instance if you want per-call temperature
        llm = self.client.bind(temperature=temperature)
        response = llm.invoke(messages)
        return response.content