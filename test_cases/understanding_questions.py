from src.model import get_model_response
import json

understanding_questions = ['Give two examples of harm caused by AI hallucinations.',
                           "Why would a company use RAG for its chatbot?",
                           'What are the benefits of RAG besides reducing hallucinations?',
                           'What two components did the 2020 paper combine?']

def test():
    responses = {}
    for question in understanding_questions:
        responses[question] = get_model_response(question)
    with open("test_cases/test_answers/understanding_answers.json",'w') as f:
        json.dump(responses,f)

if __name__=='__main__':
    test()