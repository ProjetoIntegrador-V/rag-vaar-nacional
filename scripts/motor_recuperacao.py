from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
from qwen import QwenEmbedder
from bm25 import tokenize_pt
from collections import Counter

class MotorRecuperacaoVAAR:
    def __init__(self, caminho_qdrant="./banco_qdrant_local"):
        # 1. Conexão Local: Atende ao requisito de rodar offline e sem Docker
        self.banco = QdrantClient(path=caminho_qdrant)
        
        # 2. Inicializa o Qwen3-Embedding-0.6B
        self.motor_denso = QwenEmbedder()
        
        # 3. Inicializa o Reranker Cross-Encoder. 
        # A escolha de modelos M3 ou BERTimbau garante compreensão profunda 
        # do vocabulário e gramática nativa do domínio jurídico
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

    def buscar_legislacao(self, pergunta_original, texto_expandido_hyde, top_k_busca=20, top_k_final=5):
        """
        Executa a busca multicamadas exigida pelo Tópico 5.
        Recebe a pergunta original (para BM25) e a gerada pelo HyDE (para o Qwen).
        """
        
        # ETAPA A: PREPARAÇÃO DOS VETORES
        # O vetor denso DEVE usar encode_queries() para aplicar o prefixo de instrução simétrica
        vetor_semantico = self.motor_denso.encode_queries([texto_expandido_hyde])[0]
        
        # O vetor esparso extrai as âncoras exatas (ex: "3106200", "art. 14")
        tokens = tokenize_pt(pergunta_original)
        frequencia_tokens = Counter(tokens)
        
        # ETAPA B: BUSCA HÍBRIDA NATIVA (Reciprocal Rank Fusion)
        # O Qdrant resolve a fusão RRF e aplica seu próprio cálculo IDF[cite: 3, 4]
        resultados_brutos = self.banco.query_points(
            collection_name="vaar",
            prefetch=[
                models.Prefetch(
                    query=vetor_semantico.tolist(),
                    using="denso",
                    limit=top_k_busca
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(frequencia_tokens.keys()),
                        values=list(frequencia_tokens.values())
                    ),
                    using="esparso",
                    limit=top_k_busca
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k_busca
        )

        # ETAPA C: CROSS-ENCODER RERANKING
        # Avalia a combinação exata entre a pergunta original e os Top 20 chunks do banco[cite: 7]
        pares_para_avaliacao = [
            [pergunta_original, hit.payload.get("texto", "")] 
            for hit in resultados_brutos
        ]
        
        scores_rerank = self.reranker.predict(pares_para_avaliacao)
        
        # Reordena os resultados baseados na nota cirúrgica do Cross-Encoder
        documentos_ordenados = sorted(
            zip(resultados_brutos, scores_rerank),
            key=lambda par: par[1],
            reverse=True
        )

        # ETAPA D: HIERARQUIA DE CONTEXTO (Small-to-Big Retrieval)
        # Extrai os metadados dos Top 5 finais. Se você indexou as leis com a técnica parent-document,
        # o payload retornado deve conter a 'Seção' ou 'Artigo completo' em vez do chunk solto[cite: 7]
        contexto_final_estruturado = []
        for hit, score in documentos_ordenados[:top_k_final]:
            contexto_final_estruturado.append({
                "titulo": hit.payload.get("titulo"),
                "ano": hit.payload.get("ano"),
                "texto": hit.payload.get("texto"), # Este texto será consumido pela Parte 6
                "score_rerank": float(score)
            })

        return contexto_final_estruturado
