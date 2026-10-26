"""Render the executed notebooks + written answers into one professional A4 report (HTML -> PDF via Edge)."""
import base64
import datetime as dt
import html
import json
import os
import re
import subprocess

import mistune
import nbformat
import pandas as pd
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
REPORT_DIR = os.path.join(ROOT, "report")
AUTHOR = "Guntuku Hariprasad"
PDF_NAME = "Twitter_NER_GuntukuHariprasad.pdf"

md = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def render_markdown(text):
    return md(text)


def render_output(o):
    t = o.get("output_type")
    if t == "stream":
        if o.get("name") == "stderr":
            return ""
        txt = ANSI.sub("", o["text"])
        txt = "\n".join(l for l in txt.split("\n") if not l.startswith("WARNING:tensorflow")).strip("\n").rstrip()
        return f'<pre class="out">{html.escape(txt)}</pre>' if txt else ""
    if t in ("execute_result", "display_data"):
        d = o.get("data", {})
        if "image/png" in d:
            return f'<div class="fig"><img src="data:image/png;base64,{d["image/png"]}"/></div>'
        if "text/html" in d:
            h = d["text/html"]
            h = re.sub(r"<style[\s\S]*?</style>", "", h)
            return f'<div class="tbl">{h}</div>'
        if "text/plain" in d:
            return f'<pre class="out">{html.escape(ANSI.sub("", d["text/plain"]))}</pre>'
    if t == "error":
        return f'<pre class="out err">{html.escape(o["ename"] + ": " + o["evalue"])}</pre>'
    return ""


def render_notebook(path, part_id):
    nb = nbformat.read(path, as_version=4)
    parts, n = [], 0
    for c in nb.cells:
        if c.cell_type == "markdown":
            h = render_markdown(c.source)
            # demote the notebook title (h1) to a part header, keep h2/h3 as sections
            h = re.sub(r"<h1>(.*?)</h1>", r'<h1 class="part" id="%s">\1</h1>' % part_id, h, count=1)
            parts.append(f'<div class="md">{h}</div>')
        elif c.cell_type == "code":
            n += 1
            code = highlight(c.source, PythonLexer(), HtmlFormatter(nowrap=True))
            outs = "".join(render_output(o) for o in c.get("outputs", []))
            parts.append(f'<div class="cell"><div class="cin"><span class="num">In [{n}]</span>'
                         f'<pre class="code">{code}</pre></div>{outs}</div>')
    return "\n".join(parts)


def fill(text, values):
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), m.group(0))), text)


def main():
    lstm = json.load(open(os.path.join(OUT, "lstm_best.json")))
    bert = json.load(open(os.path.join(OUT, "bert_results.json")))
    lstm_exp = pd.read_csv(os.path.join(OUT, "lstm_experiments.csv"), index_col=0)
    bert_exp = pd.DataFrame(bert["experiments"]).set_index("experiment")

    les, bes = lstm["early_stopping"], bert["early_stopping"]
    k_with, k_without = list(les["val_F1"].keys())
    b_with, b_without = list(bes["val_F1"].keys())
    v = {
        "lstm_f1": f"{lstm['test_F1']:.3f}", "bert_f1": f"{bert['test_F1']:.3f}",
        "lstm_es_epochs": int(les["epochs_run"][k_with]), "lstm_es_val": f"{les['val_F1'][k_with]:.3f}",
        "lstm_es_test": f"{les['test_F1'][k_with]:.3f}", "lstm_noes_val": f"{les['val_F1'][k_without]:.3f}",
        "lstm_noes_test": f"{les['test_F1'][k_without]:.3f}",
        "bert_es_best": int(bes["best_epoch"][b_with]), "bert_es_epochs": int(bes["epochs_run"][b_with]),
        "bert_es_test": f"{bes['test_F1'][b_with]:.3f}", "bert_es_val": f"{bes['val_F1'][b_with]:.3f}",
        "bert_noes_test": f"{bes['test_F1'][b_without]:.3f}", "bert_noes_val": f"{bes['val_F1'][b_without]:.3f}",
        "lstm_p": f"{lstm['test_P']:.3f}", "lstm_r": f"{lstm['test_R']:.3f}",
        "bert_p": f"{bert['test_P']:.3f}", "bert_r": f"{bert['test_R']:.3f}",
        "gain": f"{100 * (bert['test_F1'] - lstm['test_F1']):.1f}",
        "n_exp": len(lstm_exp) + len(bert_exp) + 1,          # +1: BiLSTM-CRF retrained without early stopping
    }
    v["bert_max_epochs"] = int(bes["epochs_run"][b_without])
    lt, lnt = les["test_F1"][k_with], les["test_F1"][k_without]
    v["lstm_es_comment"] = (
        f"For the BiLSTM-CRF, the fixed 40-epoch model scored {lnt:.3f} test F1 against {lt:.3f} with early stopping, "
        f"even though its validation F1 was lower." if lnt > lt else
        f"For the BiLSTM-CRF, early stopping gave {lt:.3f} test F1 against {lnt:.3f} for the fixed 40-epoch model.")
    bt, bnt = bes["test_F1"][b_with], bes["test_F1"][b_without]
    if v["bert_es_best"] == v["bert_max_epochs"]:
        v["bert_es_comment"] = (
            f"For BERT, the situation is different. With the linear learning-rate decay, validation F1 kept improving "
            f"through all {v['bert_max_epochs']} epochs, so early stopping never triggered and both settings produce the "
            f"same model. Training loss fell steadily while validation loss flattened, a sign that more epochs would "
            f"soon start to over-fit.")
    else:
        v["bert_es_comment"] = (
            f"For BERT, validation F1 peaked at epoch {v['bert_es_best']}, and early stopping (patience 2) would have "
            f"halted after epoch {v['bert_es_epochs']} of {v['bert_max_epochs']}. The restored best epoch scored "
            f"{bt:.3f} test F1, against {bnt:.3f} for the last epoch.")
    answers = render_markdown(fill(open(os.path.join(REPORT_DIR, "answers.md"), encoding="utf-8").read(), v))
    extra = {k: render_markdown(fill(open(os.path.join(REPORT_DIR, f"{k}.md"), encoding="utf-8").read(), v))
             for k in ["summary", "conclusion"]}

    def table(df, cols, fmt, caption, selected):
        rows = "".join(f'<tr class="{"sel" if i == selected else ""}"><td>' + html.escape(str(i)) +
                       (" &#9733; selected" if i == selected else "") + "</td>" +
                       "".join(f"<td>{fmt.get(c, '{}').format(r[c])}</td>" for c in cols) + "</tr>"
                       for i, r in df.iterrows())
        head = "".join(f"<th>{c}</th>" for c in cols)
        return (f'<h3>{caption}</h3><table class="data"><thead><tr><th>experiment</th>{head}</tr></thead>'
                f'<tbody>{rows}</tbody></table>')

    cols = ["epochs_run", "val_F1", "test_P", "test_R", "test_F1"]
    f3 = {c: "{:.3f}" for c in ["val_F1", "test_P", "test_R", "test_F1"]} | {"epochs_run": "{:.0f}"}
    exp_tables = (table(lstm_exp, cols, f3, "Part A — BiLSTM experiments (E1–E12)", lstm["experiment"]) +
                  table(bert_exp, cols, f3, "Part B — BERT fine-tuning experiments (B1–B4)", bert["selected"]))

    body = open(os.path.join(REPORT_DIR, "template.html"), encoding="utf-8").read()
    body = fill(body, {
        **v, "author": AUTHOR, "date": dt.date.today().strftime("%d %B %Y"),
        "lstm_name": html.escape(lstm["experiment"]), "bert_name": html.escape(bert["selected"]),
        "summary": extra["summary"], "conclusion": extra["conclusion"], "exp_tables": exp_tables,
        "part_a": render_notebook(os.path.join(ROOT, "notebooks", "01_eda_bilstm_crf.ipynb"), "part-a"),
        "part_b": render_notebook(os.path.join(ROOT, "notebooks", "02_bert_ner.ipynb"), "part-b"),
        "answers": answers, "pygments_css": HtmlFormatter(style="friendly").get_style_defs(".code"),
    })
    html_path = os.path.join(REPORT_DIR, "report.html")
    open(html_path, "w", encoding="utf-8").write(body)

    pdf_path = os.path.join(ROOT, PDF_NAME)
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                    f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace("\\", "/")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", html_path, "and", pdf_path, f"({os.path.getsize(pdf_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
