def roteador_de_intencao(pergunta_usuario, cliente_llm):
    prompt_roteador = f"""
    Você é um classificador binário. O usuário fará uma pergunta. 
    Responda APENAS 'SIM' se a pergunta for relacionada à legislação do Fundeb, 
    repasse de verbas, MEC, INEP ou VAAR. 
    Responda APENAS 'NÃO' para qualquer outro assunto.
    Pergunta: {pergunta_usuario}
    """
    
    resposta = cliente_llm.gerar_texto(prompt_roteador).strip().upper()
    
    if "NÃO" in resposta:
        return {"status": "descartado", "mensagem": "Pergunta fora do escopo jurídico-educacional do sistema."}
    return {"status": "aprovado", "pergunta": pergunta_usuario}
