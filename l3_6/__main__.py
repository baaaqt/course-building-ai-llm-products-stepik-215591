from pprint import pprint
from .client import OLlamaClient

if __name__ == "__main__":
    client = OLlamaClient("http://localhost:11434")
    pprint(client.get_models())
    pprint(client.answer("gemma3:1b", "How are you?"))
    pprint(list(client.stream_answer("gemma3:1b", "How are you?")))
