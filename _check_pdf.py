import fitz

d = fitz.open('data/input_pdfs/odu-12.pdf')
print(f'Pages: {len(d)}')
print(f'Size: {4801574 / 1024 / 1024:.1f} MB')

# Check text content on several pages
total_text_blocks = 0
total_img_blocks = 0
pages_with_text = 0
pages_with_images = 0

for pn in range(len(d)):
    pg = d[pn]
    blocks = pg.get_text('blocks')
    imgs = pg.get_images()
    txt = sum(1 for b in blocks if b[6] == 0)
    img_b = sum(1 for b in blocks if b[6] == 1)
    total_text_blocks += txt
    total_img_blocks += img_b
    if txt > 0:
        pages_with_text += 1
    if len(imgs) > 0 or img_b > 0:
        pages_with_images += 1

print(f'\nPages with text: {pages_with_text}/{len(d)}')
print(f'Pages with images: {pages_with_images}/{len(d)}')
print(f'Total text blocks: {total_text_blocks}')
print(f'Total image blocks: {total_img_blocks}')

# Sample text from a few pages
for pn in [5, 20, 50, 100, 150]:
    if pn < len(d):
        pg = d[pn]
        text = pg.get_text()
        imgs = pg.get_images()
        preview = text[:200].replace('\n', ' ').strip()
        print(f'\np.{pn+1} ({len(imgs)} imgs): {preview}')

d.close()
