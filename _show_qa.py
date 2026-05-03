import json

nb = json.load(open('notebooks/odu_arnold_result.ipynb', 'r', encoding='utf-8'))

def get_stdout(cell_idx):
    c = nb['cells'][cell_idx]
    text = []
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream' and o.get('name') == 'stdout':
            text.extend(o.get('text', []))
    return ''.join(text)

print('=== Q1 (общий) ===')
print(get_stdout(24))
print('\n\n=== Q2 (общий) ===')
print(get_stdout(25))
print('\n\n=== Q3 (тема: теорема) ===')
print(get_stdout(27))
print('\n\n=== Q4 (тема: фазовое пространство) ===')
print(get_stdout(28))
