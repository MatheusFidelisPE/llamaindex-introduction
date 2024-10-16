from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=120)
quest = "Quem era o presidente da Rússia em 2015?"
print(f'Pergunta: {quest}')
resp = llm.complete(quest)
print(resp)