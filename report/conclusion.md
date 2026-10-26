## 7.1 Conclusions

* A complete NER pipeline was built for noisy Twitter text: defensive CoNLL parsing and EDA; normalisation and a
  stratified train / validation split; two tokenization pipelines (TensorFlow word + char tokenizers, BERT WordPiece
  with label alignment); two model families; {{n_exp}} training runs; and word-level evaluation with seqeval.
* **BiLSTM-CRF** (TensorFlow 2.15, Word2Vec / GloVe-Twitter embeddings, char-CNN, custom CRF): test F1 **{{lstm_f1}}**.
  The experiment ladder shows the value of each component, and pre-trained embeddings give the largest gain.
* **BERT (`bert-base-uncased`)** fine-tuned with AdamW, warm-up and early stopping: test F1 **{{bert_f1}}**, the best
  model. It was saved with `save_pretrained`, reloaded, and verified to reproduce its predictions.
* For context, the winning system of the WNUT-16 shared task scored ≈ 0.52 F1, and Twitter-pre-trained transformers
  such as BERTweet report results in the low 0.50s. WNUT-16 is a hard benchmark: novel entities and noisy text keep
  scores far below those on news-domain data such as CoNLL-2003, where F1 exceeds 0.90.

## 7.2 Limitations

* **Small, shifted data.** There are only 2.4k training tweets and 240 validation tweets, so model selection is noisy
  and the test entity mix differs from training.
* **Casing is lost** in `bert-base-uncased`, although 83% of entity tokens are capitalised. A cased model keeps this cue.
* **CPU-only training** limited the BERT hyper-parameter grid to 4 runs and a maximum of 6 epochs.

## 7.3 Recommendations & next steps

| Priority | Action | Expected impact |
|---|---|---|
| 1 | Switch to a **Twitter-pre-trained, cased** transformer (BERTweet, Twitter-RoBERTa) | Largest single gain on WNUT data |
| 2 | **Add a CRF head** on top of the transformer, or use BIOES tags | Cleaner boundaries for multi-token entities |
| 3 | **Active learning** on rare types (`movie`, `tvshow`, `product`) + weak supervision from gazetteers (Wikidata) | Recall on rare / novel entities |
| 4 | k-fold cross-validation and multiple seeds | Reliable model selection |
| 5 | Distil to DistilBERT / ONNX-quantise; serve with FastAPI in Docker on Kubernetes with autoscaling | Production latency and cost at 6k tweets/s |
| 6 | Entity linking + per-entity time series → trend detection dashboard | Directly delivers the business goal |

## 7.4 Reproducibility

| Item | Value |
|---|---|
| Environment | Python 3.11, **TensorFlow 2.15.0** (tf-keras 2.15), transformers 4.40.2, gensim 4.3.3, seqeval 1.2.2; CPU (Intel i5, 8 GB RAM) |
| Randomness | `SEED = 42` for Python, NumPy and TensorFlow; stratified 90/10 split of the training file; test file untouched |
| Metric | seqeval entity-level micro P/R/F1 (CoNLL exact match) on the original word tokens |
| Code | `notebooks/01_eda_bilstm_crf.ipynb` (Part A) and `notebooks/02_bert_ner.ipynb` (Part B); this report is generated from the executed notebooks, so every number and figure is a real output |
| Artefacts | `models/bilstm_crf/` (weights, tokenizers, config) and `models/bert_ner/` (`save_pretrained` model + tokenizer) |
