from src.model import get_model_response
import json

unanswerable_questions = ["Who wrote the 2020 paper that introduced RAG?",
                          "Which company developed RAG?",
                          "What vector database is used in RAG?",
                          'How much does it cost to retrain an LLM?',
                          'What is the accuracy improvement of RAG over a standard LLM?']

def test():
    responses = {}
    for question in unanswerable_questions:
        responses[question] = get_model_response(question)

    with open("test_cases/test_answers/unanswerable_answers.json",'w') as f:
        json.dump(responses,f)

if __name__=='__main__':
    test()