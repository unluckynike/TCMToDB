import csv, sys, collections, re

def load(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

base = sys.argv[1]
formula = load(f"{base}/TCMTox_formula.csv")
herb = load(f"{base}/TCMTox_herb.csv")
ing = load(f"{base}/TCMTox_ingredient.csv")
target = load(f"{base}/TCMTox_target.csv")

print(f"formula={len(formula)} herb={len(herb)} ingredient={len(ing)} target={len(target)}")
print("formula cols:", list(formula[0].keys()))
print("herb cols:", list(herb[0].keys()))
print("ingredient cols:", list(ing[0].keys()))
print("target cols:", list(target[0].keys()))

# Herb toxicity field
tox = collections.Counter((h.get('Toxicity') or '').strip() for h in herb)
print("\n=== Herb Toxicity values ===")
for k, v in tox.most_common(15):
    print(f"  {v:4d}  {k!r}")

# Target Toxicity values
ttox = collections.Counter((t.get('Toxicity') or '').strip() for t in target)
print("\n=== Target Toxicity values (top 15) ===")
for k, v in ttox.most_common(15):
    print(f"  {v:4d}  {k!r}")

# Formula: parse Prescription to extract herb names
n_with_rx = sum(1 for f in formula if (f.get('Prescription') or '').strip())
herbs_in_rx = set()
n_parse = 0
pat = re.compile(r'([A-Za-z][A-Za-z \-]*?)(?:\s*\d+(?:\.\d+)?\s*[a-zA-Z]+)?(?:,|$)')
for f in formula:
    rx = f.get('Prescription') or ''
    if rx:
        n_parse += 1
        for part in rx.split(','):
            m = re.match(r'\s*([A-Za-z][A-Za-z \-]*?)\s*\d', part)
            if m:
                herbs_in_rx.add(m.group(1).strip().lower())
print(f"\nformulas with Prescription: {n_parse}/{len(formula)}")
print(f"distinct herb tokens parsed from prescriptions: {len(herbs_in_rx)}")
herb_pinyin = {h['Herb_pinyin_name'].strip().lower() for h in herb if h.get('Herb_pinyin_name')}
overlap = herbs_in_rx & herb_pinyin
print(f"overlap with the 229 toxic herbs: {len(overlap)}")

# Ingredient: SMILES coverage
n_smiles = sum(1 for i in ing if (i.get('Simles') or '').strip())
n_cas = sum(1 for i in ing if (i.get('CAS_ID') or '').strip())
n_pub = sum(1 for i in ing if (i.get('PubChem_ID') or '').strip())
n_uniq_names = len({i['Name'] for i in ing if i.get('Name')})
print(f"\ningredient: unique names={n_uniq_names}, with SMILES={n_smiles}, with CAS={n_cas}, with PubChem={n_pub}")

# Target: UniProt coverage
n_uniprot = sum(1 for t in target if (t.get('UniProt_ID') or '').strip())
n_seq = sum(1 for t in target if (t.get('Sequence') or '').strip())
n_uniq_genes = len({t['Gene_name'] for t in target if t.get('Gene_name')})
print(f"target: unique genes={n_uniq_genes}, with UniProt={n_uniprot}, with Sequence={n_seq}")

# Formula fields coverage
for col in ['Type','Dosage','Diseases','Syndromes','Efficacy']:
    n = sum(1 for f in formula if (f.get(col) or '').strip())
    print(f"formula.{col}: {n}/{len(formula)}")
