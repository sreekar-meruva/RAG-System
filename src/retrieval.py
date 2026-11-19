import os
from dotenv import load_dotenv
from typing import List
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

load_dotenv()

transformer = os.getenv("EMBEDDING_MODEL")
embed_model = SentenceTransformer(transformer)

crossEncoder = os.getenv("ENCODER_MODEL")
encoder_model = CrossEncoder(crossEncoder)

def get_embeddings(text: str):
    try:
        embedding = embed_model.encode(text, normalize_embeddings=True)
    except Exception as ex:
        raise RuntimeError(f"Unable to generate embeddings due to {ex}")
    return embedding

def reranker(query: str, texts:List[str]):
    try:
        encoder_input = [(query, text) for text in texts]
        sim_scores = encoder_model.predict(encoder_input)
        top_indices = np.argsort(sim_scores)[::-1]
        topk = top_indices[:3]
        top_texts = [texts[idx] for idx in topk]
    except Exception as ex:
        raise RuntimeError(f"Unable to generate encoder similarity scores due to {ex}")
    return top_texts

def getLexicalTokens(text: str):
    STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","of","to","in","on","for",
    "and","or","by","with","as","at","from","that","this","it","its","what","which",
    "who","when","how","why","does","do","did","can","into","than","their","they"
    }
    words = re.findall(r'[a-z0-9]+', text.lower())
    tokens = [word for word in words if word not in STOP_WORDS]
    return tokens

def getLexicalRanking(query_tokens: List[str], chunk_tokens: List[List[str]]):
    bm25 = BM25Okapi(chunk_tokens)
    text_scores = bm25.get_scores(query_tokens)
    return np.argsort(text_scores)[::-1]

def get_rrf_ranking(rankings: List[List[int]]):
    total_chunks = len(rankings[0])
    rrf_scores = [0]*total_chunks
    for ranking in rankings:
        for position, chunk_idx in enumerate(ranking):
            rrf_scores[chunk_idx]+=(1/(60+position+1))
    rrf_ranking = np.argsort(rrf_scores)[::-1]
    return rrf_ranking

def get_relevant_context(query: str):
    query_embedding = get_embeddings(query)
    with open(r"content_info.json", encoding='utf-8', mode='r') as f:
        content = json.load(f)

    sim_scores = []
    texts = []
    chunk_lexical_tokens = []
    for chunk in content:
        content_embedding = np.array(chunk['embedding'], dtype=np.float32)
        texts.append(chunk['text'])
        chunk_lexical_tokens.append(getLexicalTokens(chunk['text']))
        sim_scores.append(cosine_similarity([query_embedding],[content_embedding])[0][0])

    query_lexical_tokens = getLexicalTokens(query)
    bm25_ranking = getLexicalRanking(query_lexical_tokens, chunk_lexical_tokens)

    embed_ranking = np.argsort(sim_scores)[::-1]
    rrf_ranking = get_rrf_ranking([bm25_ranking, embed_ranking])

    top_k = rrf_ranking[:5]
    top_k_texts = [texts[indx] for indx in top_k]
    
    top_texts = reranker(query, top_k_texts)
    context = "\n".join(top_texts)
    return context

# question = "What kind of memory did the original RAG paper add to the language model?"
# print(get_relevant_context(question))