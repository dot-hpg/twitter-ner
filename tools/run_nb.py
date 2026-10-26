"""Build a notebook from its percent-format source in src/ and execute it in place (notebooks/)."""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

from py2nb import convert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

name = sys.argv[1]                                   # e.g. 01_eda_bilstm_crf
src, dst = os.path.join(ROOT, "src", name + ".py"), os.path.join(ROOT, "notebooks", name + ".ipynb")
os.makedirs(os.path.dirname(dst), exist_ok=True)
convert(src, dst)
nb = nbformat.read(dst, as_version=4)
t0 = time.time()
client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(dst)}})
try:
    client.execute()
except Exception as e:
    failed = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and
              any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    print(f"FAILED in cell {failed}: {type(e).__name__}")
    if failed:
        print(nb.cells[failed[0]].source[:600])
        err = [o for o in nb.cells[failed[0]].outputs if o.get("output_type") == "error"][0]
        print(err["ename"], err["evalue"])
    raise SystemExit(1)
finally:
    nbformat.write(nb, dst)
    print(f"{name}: finished in {(time.time() - t0) / 60:.1f} min -> {dst}")
