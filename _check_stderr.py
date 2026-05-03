import json, sys

nb_path = sys.argv[1]
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    outputs = c.get('outputs', [])
    for o in outputs:
        if o.get('output_type') == 'stream' and o.get('name') == 'stderr':
            text = ''.join(o.get('text', []))
            if text.strip():
                src = c.get('source', [''])[0].strip()[:50]
                print(f'Cell [{i}] ({src}):')
                print(f'  STDERR: {text[:300]}')
                print()
