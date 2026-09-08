def gerar_resposta_final(pergunta_usuario, documentos_recuperados, cliente_llm):
    # Formata os documentos recuperados (extraídos da estrutura JSON) para o prompt
    contexto_formatado = "\n\n".join(
        f"Documento: {doc['titulo']} ({doc['ano']})\nTrecho: {doc['texto']}" 
        for doc in documentos_recuperados
    )
    
    prompt_geracao = f"""
    Você é um assistente técnico especialista em Fundeb. Responda à pergunta do usuário 
    utilizando EXCLUSIVAMENTE os trechos de normas oficiais fornecidos abaixo.
    Se a resposta não estiver no contexto, responda: "A legislação recuperada não contém essa informação."
    Ao final da resposta, cite o nome do documento utilizado.
    
    Contexto Oficial:
    {contexto_formatado}
    
    Pergunta: {pergunta_usuario}
    """
    
    resposta_final = cliente_llm.gerar_texto(prompt_geracao)
    return resposta_final
