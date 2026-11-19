from src.model import get_model_response
import json

reworded_questions = ['How does RAG save money?',
                      'How can users check whether a RAG answer is correct?',
                      'What kind of memory did the original RAG paper add to the language model?',
                      'What problem with chatbots does RAG help reduce?'
                      ]

def test():
    responses = {}
    for question in reworded_questions:
        responses[question] = get_model_response(question)
    with open("test_cases/test_answers/reworded_answers.json",'w') as f:
        json.dump(responses,f)

if __name__=='__main__':
    test()