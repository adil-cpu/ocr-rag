import json

nb = json.load(open('notebooks/odu_arnold_result.ipynb', 'r', encoding='utf-8'))

def get_stdout(cell_idx):
    c = nb['cells'][cell_idx]
    text = []
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream' and o.get('name') == 'stdout':
            text.extend(o.get('text', []))
    return ''.join(text)

# Setup
print('=== [2] SETUP ===')
print(get_stdout(2))
print('\n=== [4] BLOCKS ===')
print(get_stdout(4))
print('\n=== [8] IMAGES ===')
print(get_stdout(8))
print('\n=== [9] CLIP ===')
print(get_stdout(9))
print('\n=== [11] CHARTS ===')
print(get_stdout(11)[:800])
print('\n=== [15] MARKDOWN ===')
print(get_stdout(15))
print('\n=== [19] EMBEDDINGS ===')
print(get_stdout(19))
