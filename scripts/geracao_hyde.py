def gerar_documento_hyde(pergunta_aprovada, cliente_llm):
    prompt_hyde = f"""
    Escreva um parágrafo imitando a linguagem formal de uma Nota Técnica do Inep 
    ou legislação governamental que responderia à seguinte pergunta. 
    Concentre-se em utilizar siglas institucionais e jargões técnicos. Não invente números, 
    mas simule a estrutura normativa.
    Pergunta: {pergunta_aprovada}
    """
    
    documento_hipotetico = cliente_llm.gerar_texto(prompt_hyde)
    return documento_hipotetico

# O output desta função será transformado em vetor pelo Qwen-Embedding-0.6B 
# e enviado ao Qdrant para a busca híbrida (Parte 5).
