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

direct_questions = ['When was RAG first proposed?',
             'What does RAG stand for?',
             'Which publication described RAG as blending the LLM process with a web search or document look-up?',
             'Where does RAG pull relevant text from?']

reworded_questions = ['How does RAG save money?',
                      'How can users check whether a RAG answer is correct?',
                      'What kind of memory did the original RAG paper add to the language model?',
                      'What problem with chatbots does RAG help reduce?'
                      ]

understanding_questions = ['Give two examples of harm caused by AI hallucinations.',
                           "Why would a company use RAG for its chatbot?",
                           'What are the benefits of RAG besides reducing hallucinations?',
                           'What two components did the 2020 paper combine?']

unanswerable_questions = ["Who wrote the 2020 paper that introduced RAG?",
                          "Which company developed RAG?",
                          "What vector database is used in RAG?",
                          'How much does it cost to retrain an LLM?',
                          'What is the accuracy improvement of RAG over a standard LLM?']

false_premises = ['Why was RAG first proposed in 2018?',
                  'Why does RAG require retraining the LLM every time new data is added?',
                  'Which article by Wired described RAG as blending LLMs with web search?']

# total_answers = {}

# print("Direct question answers: ")
# answers = []
# for question in direct_questions:
#     messages = ask_model(question)
#     answers.append(messages[-1]['content'])
# total_answers['direct_answers'] = answers

# print("Reworded question answers: ")
# answers = []
# for question in reworded_questions:
#     messages = ask_model(question)
#     answers.append(messages[-1]['content'])
# total_answers['reworded_answers'] = answers

# print("Understanding question answers: ")
# answers=[]
# for question in understanding_questions:
#     messages = ask_model(question)
#     answers.append(messages[-1]['content'])
# total_answers['understanding_answers'] = answers

# print("Unanswerable question answers: ")
# answers=[]
# for question in unanswerable_questions:
#     messages = ask_model(question)
#     answers.append(messages[-1]['content'])
# total_answers['unanswerable_answers']=answers

# print("False premises questions: ")
# answers = []
# for question in false_premises:
#     messages = ask_model(question)
#     answers.append(messages[-1]['content'])
# total_answers['false_premises']=answers

# with open("answers.json",mode='w') as f:
#     json.dump(total_answers,f)

# question = 'What problem with chatbots does RAG help reduce?'
# messages = ask_model(question)
# answer = messages[-1]['content']
# if answer == "I don't know based on the provided documents.":
#     messages = retry_model(question)
# print(messages[-1]['content'])