# combine_data.py — Merge two or more MasterSOM .data files into one combined file
#
# Reads multiple YAML-header .data files (as written by random_search.py), verifies that
# all files share the same variable list, stacks their data rows, deduplicates, and writes
# a single combined .data file ready for MasterSOM.java. Edit INPUT_DATA_FILES and
# OUTPUT_DATA directly in this script before running.
#
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "outputs", "som")

INPUT_DATA_FILES = [
    os.path.join(OUTDIR, "round2_seed42_43.data"),
    os.path.join(OUTDIR, "round3_seed44_500.data"),
]

OUTPUT_DATA = os.path.join(OUTDIR, "round3_seed42.data")

# ── Parse a .data file ────────────────────────────────────────────────────────

def parse_data_file(path):
    """Return (variables, data_rows) for the given .data file."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    dashes       = [i for i, l in enumerate(lines) if l.strip() == "---"]
    header_lines = lines[dashes[0]+1 : dashes[1]]
    data_lines   = [l.strip() for l in lines[dashes[1]+1:] if l.strip()]

    variables = []
    in_vars   = False
    for l in header_lines:
        s = l.strip()
        if s.startswith("variables:"):
            in_vars = True
        elif in_vars and s.startswith("- "):
            variables.append(s[2:].strip())
        elif in_vars and not s.startswith("-"):
            in_vars = False

    return variables, data_lines

# ── Load all files ────────────────────────────────────────────────────────────

all_variables = None
all_rows      = []

for path in INPUT_DATA_FILES:
    if not os.path.exists(path):
        print(f"WARNING: Not found, skipping: {os.path.basename(path)}")
        continue

    variables, rows = parse_data_file(path)

    if all_variables is None:
        all_variables = variables
    elif variables != all_variables:
        print(f"WARNING: Column mismatch in {os.path.basename(path)} — skipping")
        print(f"   Expected : {all_variables}")
        print(f"   Got      : {variables}")
        continue

    all_rows.extend(rows)
    print(f"  Loaded {os.path.basename(path):55s}  {len(rows):5d} rows")

# ── Deduplicate ───────────────────────────────────────────────────────────────

before   = len(all_rows)
all_rows = list(dict.fromkeys(all_rows))   # preserves order, removes duplicates
after    = len(all_rows)

print(f"\nTotal rows : {before}")
print(f"Duplicates : {before - after}")
print(f"Unique rows: {after}")
print(f"Variables  : {all_variables}")

# ── Write combined .data file ─────────────────────────────────────────────────

n_vars    = len(all_variables)
n_inputs  = 20
n_outputs = 0

header  = "---\n"
header += f'name: "SOM {os.path.basename(OUTPUT_DATA)[:-5]} — {after} samples"\n'
header += f"description: Factor-based strategy search combined\n"
header += "variables:\n"
for v in all_variables:
    header += f"    - {v}\n"
header += f"inputs: {n_inputs}\n"
header += f"outputs: {n_outputs}\n"
header += "---\n"

with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
    f.write(header)
    for row in all_rows:
        f.write(row + "\n")

print(f"\nWritten : {OUTPUT_DATA}")
print(f"   Columns : {n_vars}  (inputs={n_inputs}, outputs={n_outputs})")
print(f"   Rows    : {after}")
print(f"\nNext step: set dataset path in MasterSOM.java to:")
print(f"  {OUTPUT_DATA}")
