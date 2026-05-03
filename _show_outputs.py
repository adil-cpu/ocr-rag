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

# Blocks
print('\n=== [4] BLOCKS ===')
print(get_stdout(4))

# Block examples
print('\n=== [6] BLOCK EXAMPLES ===')
print(get_stdout(6)[:600])

# Image extraction
print('\n=== [8] IMAGE EXTRACTION ===')
print(get_stdout(8))

# CLIP
print('\n=== [9] CLIP CLASSIFICATION ===')
print(get_stdout(9))

# Charts
print('\n=== [11] CHART TRANSCRIPTION ===')
print(get_stdout(11))

# Markdown
print('\n=== [15] MARKDOWN ===')
print(get_stdout(15))

# Chunk example
print('\n=== [16] CHUNK EXAMPLE ===')
print(get_stdout(16)[:400])

# Embeddings
print('\n=== [19] EMBEDDINGS ===')
print(get_stdout(19))

# Q&A - all 6
for qi, label in [(24, 'Q1 общий'), (25, 'Q2 общий'), (27, 'Q3 тема'), (28, 'Q4 тема'), (30, 'Q5 графики'), (31, 'Q6 формулы')]:
    print(f'\n=== [{qi}] {label} ===')
    print(get_stdout(qi))

# Stats
print('\n=== [33] FINAL STATS ===')
print(get_stdout(33))
