import csv, sys, collections

def load(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

base = sys.argv[1]
ing = load(f"{base}/TCMTox_ingredient.csv")
herb = load(f"{base}/TCMTox_herb.csv")

herb_ids = {h['HERB_id'] for h in herb if h.get('HERB_id')}
n_with = 0; links = set()
for i in ing:
    raw = (i.get('HERB_ID') or '').strip()
    if not raw: continue
    n_with += 1
    for tok in raw.replace(';', ' ').split():
        tok = tok.strip()
        if tok in herb_ids:
            links.add((tok, i['Name']))
print(f"ingredients with HERB_ID: {n_with}/{len(ing)}")
print(f"rebuildable herb-ingredient links: {len(links)}")
print(f"herbs covered by those links: {len({a for a,b in links})}")

# toxicity distribution of herbs
tox_map = {'Extremely Toxic':3,'Toxic':2,'toxic':2,'Slightly Toxic':1,'slightly toxic':1}
herb_tox = {}
for h in herb:
    t = (h.get('Toxicity') or '').strip()
    if t: herb_tox[h['Herb_pinyin_name']] = tox_map.get(t, 0)
# formula -> toxic herb usage
formula = load(f"{base}/TCMTox_formula.csv")
import re
pinyin_set = set(herb_tox)
n_formula_with_toxic = 0
examples = []
for f in formula:
    rx = (f.get('Prescription') or '')
    found = []
    for part in rx.split(','):
        m = re.match(r'\s*([A-Za-z][A-Za-z \-]*?)\s*\d', part)
        if m and m.group(1).strip().lower() in pinyin_set:
            found.append(m.group(1).strip())
    if found:
        n_formula_with_toxic += 1
        if len(examples) < 5: examples.append((f['Form_name'], found))
print(f"formulas containing >=1 of the 229 toxic herbs: {n_formula_with_toxic}")
print("examples:", examples[:3])
