import os
from dotenv import load_dotenv
import json
import uuid
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

transformer = os.getenv("EMBEDDING_MODEL")

model = SentenceTransformer(transformer)

with open(r"document.txt", encoding='utf-8', mode='r') as f:
    document = f.read()

paragraphs = document.split('\n')
paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
sentence_list = [re.split(r"(?<=\.)",paragraph) for paragraph in paragraphs]

for sentences in sentence_list:
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

embeddings = [model.encode(sentences, normalize_embeddings=True) for sentences in sentence_list]

document_chunks = []

try:
    for indx,embedding in enumerate(embeddings):
        chunks = []
        chunk = [sentence_list[indx][0]]
        sim_scores = []
        for idx in range(1, len(embedding)):
            prev_embedding = [embedding[idx-1]]
            curr_embedding = [embedding[idx]]
            sim_score = cosine_similarity(prev_embedding, curr_embedding)
            sim_scores.append(sim_score)
            if sim_score>=0.50:
                chunk.append(sentence_list[indx][idx])
            else:
                subparagraph = " ".join(chunk)
                if subparagraph.strip(): chunks.append(subparagraph.strip())
                chunk = [sentence_list[indx][idx]]
        if chunk:
            subparagraph = " ".join(chunk)
            if subparagraph.strip(): chunks.append(subparagraph.strip())
        document_chunks.append(chunks)
except Exception as ex:
    raise ValueError(f"Unable to chunk due to {ex}")

embed_chunks = []
for paragraph in document_chunks:
    embedding = model.encode(paragraph, normalize_embeddings=True)
    embed_chunks.append(embedding)

chunkfile = []
para_num = 1
for idx, paragraph in enumerate(document_chunks):
    for indx in range(len(paragraph)):
        chunk_data = {
            'text': paragraph[indx],
            'embedding': embed_chunks[idx][indx].tolist(),
            'chunk_id': str(uuid.uuid4()),
            'paragraph': 'Paragraph_'+str(para_num)
        }
        chunkfile.append(chunk_data)
    para_num+=1

with open("content_info.json", mode='w') as f:
    json.dump(chunkfile, f)
