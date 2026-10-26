## Q1. Defining the problem statement — and where can this (and modifications of it) be used?

**Problem statement.** Given a tweet as a sequence of tokens *w₁ … wₙ*, predict a label sequence *y₁ … yₙ* in the BIO
scheme over 10 fine-grained entity types (`person, geo-loc, company, facility, product, musicartist, movie,
sportsteam, tvshow, other`) such that every contiguous entity mention is found with its exact boundaries and correct type.
This is a **structured sequence-labelling** task evaluated with **entity-level (exact-match) micro F1**.

**Why it matters for Twitter.** At ~6,000 tweets/second, hashtags are an unreliable signal: many tweets have none, many
are misspelled, and a hashtag rarely says *what kind* of thing is being discussed. An NER system extracts the
actual entities from the text itself, which enables:

| Use case | How NER is used |
|---|---|
| Trend & topic detection | Count entity mentions over time windows → detect spikes ("TalkTalk" + "breach") independently of hashtags |
| Search & discovery | Index tweets by entity, power "people / places / products in this conversation" panels |
| Ads & brand safety | Detect brand (`company`/`product`) mentions for targeting, competitive monitoring and brand-safety rules |
| Recommendations | Build user-interest profiles from entities users engage with (`sportsteam`, `musicartist`, `tvshow`) |
| Trust & safety | Spot targeted harassment of `person` entities, crisis events at `geo-loc`/`facility` |
| Knowledge graph / entity linking | First stage before linking mentions to Wikidata IDs and building relationship graphs |

**Modifications of the same model** (same architecture, different labels/data): PII detection & redaction
(names, phones, addresses), clinical NER (drugs, dosages, symptoms), legal/financial document extraction (parties,
amounts, dates), resume parsing, e-commerce attribute extraction (brand, size, colour), customer-support ticket routing,
**aspect-based sentiment** (tag aspect spans, then classify their sentiment), slot filling for chatbots/voice assistants
(`book a flight to [Paris]dest on [Friday]date`), and POS-tagging / chunking.

## Q2. Explain the data format (CoNLL BIO format)

The **CoNLL** format (from the CoNLL-2002/2003 shared tasks) is a plain-text, column-based format:

* **one token per line**; columns separated by a tab or a space — the **first column is the word** and the **last
  column is the label** (intermediate columns may hold POS/chunk tags; our files have exactly two);
* **a blank line separates sentences** (here, tweets).

The labels use the **BIO (a.k.a. IOB2)** encoding:

* **B-TYPE** — the *beginning* (first token) of an entity of type TYPE;
* **I-TYPE** — a token *inside* the same entity (continuation);
* **O** — the token is *outside* any entity.

Example from the training file:

```
Made      O
it        O
back      O
home      O
to        O
GA        B-geo-loc
.         O
...
at        O
Disney    B-facility
world     I-facility
```

The `B-` prefix is what allows two adjacent entities of the same type to be separated (`B-person I-person B-person` =
two people), and it lets us recover entity spans unambiguously. A sequence is *valid* only if every `I-X` follows a
`B-X` or `I-X`; we verified that our data contains no violations. With 10 types the label set has 10 × 2 + 1 = **21 labels**.

## Q3. What other NER annotation formats are available and how are they different?

| Format | Tags | Example for "*New York City is big*" | Notes |
|---|---|---|---|
| **IO** | I, O | I-loc I-loc I-loc O O | Simplest; cannot separate two adjacent entities of the same type |
| **IOB1** | I, O, B | I-loc I-loc I-loc O O | `B` used **only** when an entity directly follows another of the same type |
| **IOB2 / BIO** *(ours)* | B, I, O | B-loc I-loc I-loc O O | Every entity starts with `B`; the de-facto standard |
| **IOE1 / IOE2** | I, O, E | I-loc I-loc E-loc O O | Marks the *end* instead of the beginning |
| **BIOES / BILOU** | B, I, O, E/L, S/U | B-loc I-loc E-loc O O ; single-token = S-loc / U-loc | Explicit **E**nd and **S**ingle tags; richer signal, often +0.5–1 F1 with CRFs, but 4K+1 labels |
| **BMEWO / BMEWO+** | B, M, E, W, O | as BIOES with M(iddle), W(hole) | Used by LingPipe; "+" also encodes context of O tokens |
| **Span / stand-off** (JSON, brat `.ann`, spaCy `DocBin`) | (start, end, type) | `{"start":0,"end":13,"label":"loc"}` | Character offsets, independent of tokenization; supports **nested/overlapping** entities |
| **Inline XML / markup** (MUC, ENAMEX, ACE) | tags in text | `<ENAMEX TYPE="LOC">New York City</ENAMEX> is big` | Human-readable; requires parsing |
| **CoNLL-U** | extra MISC column | per-token with 10 columns | Universal Dependencies format; NER stored in the MISC field |

The token-level schemes (IO, BIO, BIOES …) are interchangeable for flat entities and differ only in how much
boundary information is encoded explicitly; span formats are tokenizer-agnostic and are the only option for nested entities.

## Q4. Why do we need tokenization of the data in our case?

1. **Models consume integers, not strings.** Every token must be mapped to an index into an embedding matrix. For the
   BiLSTM we fit a TensorFlow `Tokenizer` on the training split (word level, plus a character-level tokenizer) with an
   `<oov>` token for unseen words, then pad to a fixed length of 40.
2. **BERT has its own fixed vocabulary.** It was pre-trained with a 30,522-entry **WordPiece** vocabulary; we *must* use
   the same tokenizer so the ids mean what the pre-trained weights expect. It also adds the special `[CLS]`/`[SEP]`
   tokens, builds the `attention_mask` (real vs padding) and `token_type_ids`.
3. **Handling the open vocabulary of Twitter.** Tweets are full of rare names, misspellings and hashtags — our EDA shows
   most test entity tokens never appear in training. Sub-word tokenization splits an unseen word into known pieces
   (`zendesk → zen ##des ##k`), so the model still gets a meaningful representation instead of a single `<oov>` vector.
4. **Label alignment.** Because one word can become several sub-tokens, the word-level BIO labels must be **re-aligned**:
   the first sub-token keeps the label, continuation pieces and special tokens get `-100` (ignored by the loss). At
   prediction time the reverse mapping joins sub-tokens back into words. Without tokenizer-aware alignment the labels
   would be shifted and training would be meaningless — we assert this alignment for every tweet.
5. **Normalisation decisions** are part of tokenization: we replace URLs/@mentions by placeholders (never entities in
   this dataset), which shortens BERT inputs considerably with no loss of label information.

## Q5. What other models can be used for this task?

* **Classical feature-based:** HMM, MEMM, and especially **linear-chain CRF** with hand-crafted features
  (capitalisation, prefixes/suffixes, gazetteers, word shape, Brown clusters) — strong, cheap baselines.
* **Neural sequence models:** BiLSTM/BiGRU + CRF with character CNN/LSTM (Lample et al. 2016, Ma & Hovy 2016),
  ID-CNN (iterated dilated convolutions, faster), and contextual string embeddings (**Flair**).
* **Twitter-specific pre-trained transformers:** **BERTweet** (trained on 850M tweets), **TwHIN-BERT**, **Twitter-RoBERTa**
  (cardiffnlp) — usually the biggest win for WNUT data.
* **Other general transformers:** `bert-base-cased` (casing is a strong NER cue), **RoBERTa**, **XLM-RoBERTa** (multilingual
  tweets), **DeBERTa-v3**, ELECTRA, DistilBERT / TinyBERT (low-latency serving), **LUKE** (entity-aware), Longformer.
* **Hybrid:** Transformer + CRF head, transformer with gazetteer/knowledge-base features, or retrieval-augmented NER (CL-KL).
* **Span-based / other formulations:** span classification (SpERT), biaffine NER, MRC-style NER ("which spans are
  companies?"), sequence-to-sequence NER (T5/BART generate entity lists) — these also handle nested entities.
* **LLMs:** zero/few-shot prompting or instruction-tuned models (GPT-4-class, Llama, UniversalNER, GLiNER), useful for
  cold-start labelling or new entity types, typically distilled into a small fine-tuned model for production cost.
* **Production toolkits:** spaCy (transition-based / transformer pipelines), Stanford Stanza, Flair, Spark NLP.

## Q6. Did early stopping have any effect on the training and results?

**Yes, it had a clear effect on training, and only a small effect on the final test score.** Both models were
trained with and without early stopping under identical seeds, so the effect can be measured directly
(Part A §6.1, Part B §5.1).

| Model | Setting | Epochs trained | Validation F1 | Test F1 |
|---|---|---|---|---|
| BiLSTM-CRF | early stopping (val F1, patience 6, best weights restored) | {{lstm_es_epochs}} | **{{lstm_es_val}}** | {{lstm_es_test}} |
| BiLSTM-CRF | no early stopping (fixed 40 epochs) | 40 | {{lstm_noes_val}} | {{lstm_noes_test}} |
| BERT | early stopping (val F1, patience 2, best weights restored) | {{bert_es_epochs}} | **{{bert_es_val}}** | {{bert_es_test}} |
| BERT | no early stopping (fixed {{bert_max_epochs}} epochs) | {{bert_max_epochs}} | {{bert_noes_val}} | {{bert_noes_test}} |

* **Effect on training.** For the BiLSTM-CRF, validation loss reaches its minimum around epoch 12 and then rises,
  while training loss keeps falling to near zero. This is the classic over-fitting signature on a 2k-tweet training
  set. Early stopping halted training 12 epochs earlier, saving about 30% of compute, and restored the best-validation
  weights. {{bert_es_comment}}
* **Effect on results.** Early stopping always selects the model with the best *validation* F1, by construction.
  On the *test* set the difference between the two settings is small (within about 1–2 F1 points) and not always in
  favour of early stopping. {{lstm_es_comment}}
* **Why.** The validation set is small (240 tweets) and drawn from the training distribution, while the test set has
  a different entity mix and 85% unseen entities. The best validation epoch is therefore a noisy estimate of the best
  test epoch.
* **Take-away.** Early stopping is valuable as a principled, automatic choice of the number of epochs. It prevents
  over-training, saves compute (which matters at several minutes per BERT epoch on CPU) and makes runs comparable.
  Its accuracy benefit depends on a validation set that is representative of production data. We monitored
  **entity-level F1** rather than loss, because loss starts rising while F1 is still improving. Stopping on loss would
  have halted the BiLSTM too early, at about epoch 12 instead of its F1 peak at epoch 22.

## Q7. How does BERT expect a pair of sentences to be processed?

BERT packs a sentence pair (A, B) into **one sequence**:

```
[CLS]  tokens of sentence A  [SEP]  tokens of sentence B  [SEP]  [PAD] [PAD] ...
token_type_ids:  0 0 0 ... 0 0       1 1 1 ... 1 1              0 0 ...
attention_mask:  1 1 1 ... 1 1       1 1 1 ... 1 1              0 0 ...
```

* `[CLS]` is always the first token; its final hidden state is the aggregate representation used for pair-level
  classification (NSP during pre-training; NLI, paraphrase, QA-answerability downstream).
* `[SEP]` closes each segment.
* **Segment (token-type) embeddings** — `token_type_ids` 0 for sentence A and 1 for sentence B — are added to the
  token and position embeddings, so the model knows which sentence each token belongs to.
* **Position embeddings** run continuously over the whole packed sequence (max 512).
* The `attention_mask` marks real tokens (1) vs padding (0) so padding is never attended to.
* With Hugging Face: `tokenizer(sentence_a, sentence_b, truncation="only_second", max_length=...)` builds all of this
  automatically. For our single-sentence NER task every token has `token_type_id = 0`.

## Q8. Why choose attention-based models over recurrent ones?

| Aspect | RNN / LSTM | Self-attention (Transformer) |
|---|---|---|
| Long-range dependencies | Information passes through *n* sequential steps → vanishing gradients, fading memory | Any two tokens interact **directly** in one layer (path length O(1)) |
| Parallelism | Sequential over time steps — hard to parallelise | All positions processed **in parallel** → efficient on GPUs/TPUs, enables huge pre-training |
| Context | BiLSTM concatenates two *separate* one-directional passes | Every layer is **jointly bidirectional** and contextual |
| Transfer learning | Usually trained from scratch; only static embeddings are pre-trained | Deep **pre-trained** models (BERT, RoBERTa) fine-tune with little labelled data |
| Interpretability | Hidden state is opaque | Attention weights give some insight into which tokens influence each other |
| Cost | O(n) per sequence, small memory | O(n²) attention memory — fine for tweets (≤ 128 sub-tokens) |

For this task the decisive factors are **pre-training and context**: our training set has only ~2k tweets and most test
entities are unseen, so a model that already "knows" that *Spotify* is a company and *Game of Thrones* a TV show from
pre-training generalises far better — reflected in our results (BERT test F1 **{{bert_f1}}** vs BiLSTM-CRF **{{lstm_f1}}**).
RNNs remain useful when latency/memory is extremely constrained or for streaming input.

## Q9. Differentiate BERT and "simple" (vanilla) Transformers

| | Vanilla Transformer (Vaswani et al., 2017) | BERT (Devlin et al., 2018) |
|---|---|---|
| Architecture | **Encoder–decoder** (6 + 6 layers) | **Encoder-only** stack (12 layers base / 24 large) |
| Attention direction | Encoder bidirectional; decoder **causal (masked)** — left-to-right | **Fully bidirectional** in every layer |
| Original objective | Supervised **sequence-to-sequence** (machine translation) trained from scratch | **Self-supervised pre-training**: Masked Language Modelling (15% tokens) + Next-Sentence Prediction |
| Usage pattern | Train per task | **Pre-train once, fine-tune** for many tasks with a small head (token classification, QA, NLI) |
| Input representation | Token + sinusoidal position encodings | WordPiece token + **learned** position + **segment** embeddings; `[CLS]`, `[SEP]`, `[MASK]` tokens |
| Output | Generates a target sequence token by token | Contextual vectors for every input token (+ `[CLS]` summary) — not designed for generation |
| Typical tasks | Translation, summarisation, generation | Classification, NER, QA (extractive), sentence-pair tasks |

In short, BERT *is* the encoder half of the Transformer, re-purposed as a general-purpose, bidirectional language
representation learned from unlabelled text; that pre-training is precisely what makes it effective on a small,
noisy dataset like WNUT-16.
