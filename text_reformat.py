import re
with open(r"document.txt", encoding='utf-8', mode='r') as file:
    info = file.read()

paragraphs = info.split('\n')

re_pattern = "\[\d+\]"

for idx,paragraph in enumerate(paragraphs):
    paragraphs[idx] = re.sub(re_pattern,"", paragraph)

formatted_paragraphs = '\n'.join(paragraphs)
formatted_paragraphs = formatted_paragraphs.strip()

try:
    with open(r"document.txt",mode='w') as f:
        f.write(formatted_paragraphs)
except Exception as ex:
    raise(f'Failed due to {ex}')