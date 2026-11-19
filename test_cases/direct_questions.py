from src.model import get_model_response
import json

direct_questions = ['When was RAG first proposed?',
             'What does RAG stand for?',
             'Which publication described RAG as blending the LLM process with a web search or document look-up?',
             'Where does RAG pull relevant text from?']

def test():
    responses = {}
    for question in direct_questions:
        responses[question] = get_model_response(question)
    with open("test_cases/test_answers/direct_answers.json",'w') as f:
        json.dump(responses, f)

if __name__=='__main__':
    test()


