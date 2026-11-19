import os
from dotenv import load_dotenv
from openai import OpenAI
from src.retrieval import get_relevant_context

load_dotenv()

LLM_URL = os.getenv("MODEL_URL")
LLM_KEY = os.getenv("MODEL_KEY")
LLM = os.getenv("MODEL")

client = OpenAI(
    api_key = LLM_KEY,
    base_url= LLM_URL
)

with open(r"document.txt", encoding='utf-8', mode='r') as f:
    content = f.read()

def ask_model(query: str):
    system_prompt = """
    You answer questions using the context provided.
    Rules:
    1. The FACTS in your answer must come from the context. Do not add examples,
    benefits, or details that are not in the context.
    2. You MAY use general knowledge to understand the question and to match it
    to the context (e.g., knowing a name refers to a publication, or that
    "looked up" means "retrieved"). Wording does not need to match exactly.
    3. First find the sentence in the context that answers the question.
    If it exists, answer from it in 1-3 sentences.
    4. If no sentence answers the specific question asked (e.g., the context
    gives a date when asked for a person), reply exactly:
    "I don't know based on the provided documents."
    5. If the question contains an assumption that the context contradicts
   (a wrong date, name, source, or claim), state the correction, e.g.:
   - "RAG was first proposed in 2020, not 2018."
   - "That description comes from Ars Technica, not Wired."
    """
    source = get_relevant_context(query)
    context = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': f'context: {source} \n\n question: {query}'
        }
    ]

    try:
        resp = client.chat.completions.create(
            model= LLM,
            temperature=0,
            max_tokens=200,
            messages= context,
            reasoning_effort='low'
        )
    except Exception as ex:
        raise RuntimeError(f"Unable to process request due to {ex}")

    response = resp.choices[0].message.content

    context.append({
        'role': 'assistant',
        'content': response
    })

    return context

def retry_model(query: str):

    sys_prompt = """Rewrite the question as a search query for retrieving documents. 
        Keep the exact information being asked for (e.g., an amount, a name, 
        a date, a number). Do not broaden or change the question. 
        Output only the query."""
    messages = [
        {
            'role': 'system',
            'content': sys_prompt
        },
        {
            'role': 'user',
            'content': f'query: {query}'
        }
    ]
    try:
        resp = client.chat.completions.create(
            model = LLM,
            temperature = 0,
            max_tokens = 200,
            messages = messages,
            reasoning_effort='medium'
        )
    except Exception as ex:
        raise RuntimeError(f"Unable to process the request due to {ex}")
    
    query = resp.choices[0].message.content
    response = ask_model(query)
    return(response)

def get_model_response(query: str):
    messages = ask_model(query)
    answer = messages[-1]['content']
    if answer == "I don't know based on the provided documents.":
        messages = retry_model(query)
        answer = messages[-1]['content']
    return(answer)