def avaliar_factualidade(resposta_gerada, documentos_recuperados, cliente_llm):
    prompt_auditoria = f"""
    Analise a resposta abaixo e compare com o contexto fornecido. 
    A resposta contém alguma informação, número ou sigla que NÃO está presente no contexto?
    Responda apenas com a nota: 
    1.0 = Totalmente ancorado no contexto.
    0.0 = Contém alucinação ou informações externas.
    
    Contexto: {documentos_recuperados}
    Resposta: {resposta_gerada}
    """
    
    score_factualidade = float(cliente_llm.gerar_texto(prompt_auditoria))
    
    # As métricas de recuperação (Recall@K, MRR) discutidas na Parte 5 devem ser agregadas 
    # a este score para compor o dashboard de qualidade do Projeto.
    return score_factualidade
