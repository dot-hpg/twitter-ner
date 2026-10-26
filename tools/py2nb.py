"""Convert a percent-format script (# %% / # %% [markdown]) into a Jupyter notebook."""
import sys
import nbformat


def convert(src, dst):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "md":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n")]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in open(src, encoding="utf-8").read().split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
