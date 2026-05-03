import json, sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/diploma_result.ipynb'
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

print(f'File: {nb_path}')
print(f'Total cells: {len(nb["cells"])}')

for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    ec = c.get('execution_count', '-')
    outputs = c.get('outputs', [])
    has_error = any(o.get('output_type') == 'error' for o in outputs)
    src = c.get('source', [])
    first_line = src[0].strip()[:60] if src else '(empty)'
    status = 'ERROR' if has_error else ('OK' if outputs else 'no-output')
    print(f'  [{i:2d}] exec={str(ec):4s} {status:10s} {first_line}')
    if has_error:
        for o in outputs:
            if o.get('output_type') == 'error':
                print(f'        >>> {o.get("ename","")}: {o.get("evalue","")[:200]}')
