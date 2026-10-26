"""Copy markdown cells from a src/*.py notebook source into its executed notebook (code cells and outputs untouched)."""
import os
import sys

import nbformat

from py2nb import convert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1]
tmp = os.path.join(ROOT, "notebooks", f"_{name}_src.ipynb")
convert(os.path.join(ROOT, "src", name + ".py"), tmp)
src = nbformat.read(tmp, as_version=4)
os.remove(tmp)
dst_path = os.path.join(ROOT, "notebooks", name + ".ipynb")
dst = nbformat.read(dst_path, as_version=4)
assert [c.cell_type for c in src.cells] == [c.cell_type for c in dst.cells], "cell structure differs"
changed = 0
for s, d in zip(src.cells, dst.cells):
    if s.cell_type == "markdown" and s.source != d.source:
        d.source = s.source; changed += 1
    elif s.cell_type == "code":
        assert s.source == d.source, "code changed since execution - re-run the notebook instead"
nbformat.write(dst, dst_path)
print(f"{name}: updated {changed} markdown cell(s)")
