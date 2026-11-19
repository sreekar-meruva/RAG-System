from src.model import get_model_response
import json

false_premises = ['Why was RAG first proposed in 2018?',
                  'Why does RAG require retraining the LLM every time new data is added?',
                  'Which article by Wired described RAG as blending LLMs with web search?']

def test():
    responses = {}
    for question in false_premises:
        responses[question] = get_model_response(question)

    with open("test_cases/test_answers/false_premise_answers.json",'w') as f:
        json.dump(responses,f)

if __name__=="__main__":
    test()