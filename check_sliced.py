import random
from pathlib import Path

root = Path('dataset/deepscore_sliced')
images = sorted((root / 'images').glob('*.jpg'))
print('total tiles:', len(images))
labels_dir = root / 'labels'
missing = []
invalid = []

sample = random.sample(images, min(5, len(images)))
for img_path in sample:
    label_path = labels_dir / (img_path.stem + '.txt')
    if not label_path.exists():
        missing.append(img_path.name)
        continue
    with label_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        invalid.append((img_path.name, 'empty label file'))
        continue
    for idx, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            invalid.append((img_path.name, f'line {idx} has {len(parts)} parts'))
            break
        cls, cx, cy, w, h = parts
        try:
            cx, cy, w, h = map(float, (cx, cy, w, h))
        except ValueError:
            invalid.append((img_path.name, f'line {idx} not numeric'))
            break
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            invalid.append((img_path.name, f'line {idx} values out of range ({cx},{cy},{w},{h})'))
            break
    else:
        print(f"Sample tile {img_path.name}: {len(lines)} boxes, first line -> {lines[0]}")

print('\nMissing labels:', len(missing))
if missing:
    print(missing[:5])
print('Invalid labels:', len(invalid))
if invalid:
    print(invalid[:5])
