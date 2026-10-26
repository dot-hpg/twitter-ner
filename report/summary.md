**Business problem.** Twitter wants to understand trends and topics directly from tweet text, without depending on
hashtags that are often missing, misspelled or misleading. The core capability is **Named Entity Recognition (NER)**:
finding every mention of a *person, geo-location, company, facility, product, music artist, movie, sports team,
TV show* or *other* entity in a tweet, with exact boundaries.

**Data.** WNUT-16 Twitter NER corpus in CoNLL/BIO format: 2,394 training tweets (46k tokens, 1,496 entities) and
3,849 test tweets (62k tokens, 3,473 entities), 10 types → 21 BIO labels. EDA shows what makes the task hard:
**90–95% of tokens are `O`**, **85% of test entities never appear in training**, the entity mix shifts between train and
test, and the text is noisy (URLs, @mentions, hashtags, slang, emoji).

**Approach.**

| | Model A — BiLSTM-CRF | Model B — BERT |
|---|---|---|
| Input | TF `Tokenizer` (words, normalised) + char-level tokenizer | WordPiece `bert-base-uncased` tokenizer, labels aligned to first sub-token |
| Representation | Word2Vec (trained) / GloVe-Twitter (2B tweets) + char-CNN | 12-layer pre-trained transformer, fine-tuned end-to-end |
| Output layer | Custom TensorFlow **CRF** with Viterbi decoding | Linear token-classification head + sub-token → word joining |
| Tuning | 12 runs: direction, CRF, embeddings, char features, units, dropout, LR, batch | 4 runs: learning rate, batch size, AdamW vs Adam, epochs; early stopping |
| Best test F1 | **{{lstm_f1}}** (P {{lstm_p}} / R {{lstm_r}}) | **{{bert_f1}}** (P {{bert_p}} / R {{bert_r}}) |

**Key findings.**

1. **Pre-trained knowledge matters most.** For the BiLSTM-CRF, swapping embeddings trained on our 2k tweets for
   GloVe-Twitter embeddings took test F1 from 0.236 to 0.412, more than every architectural change combined.
   Fine-tuned BERT adds another **+{{gain}} F1 points** on top of that.
2. **Architecture still helps.** Bidirectional context (+3 pts), a CRF output layer (+2 pts) and character features
   (+5 pts) each improved the LSTM, and the CRF guarantees valid BIO sequences.
3. **Tokenization is the critical engineering step for BERT.** WordPiece removes the out-of-vocabulary problem, but
   it splits words into sub-tokens. Labels must be re-aligned for training, and predictions joined back to words for
   evaluation. Both steps are verified with assertions on every tweet.
4. **Early stopping** prevented over-training (validation loss rises after a few epochs) and saved compute. With a
   240-tweet validation set and a shifted test distribution, its effect on test F1 was small. See Q6.
5. **Where models fail:** rare and open-ended types (`movie`, `tvshow`, `product`, `other`) and semantically ambiguous
   names (`musicartist ↔ person`, `sportsteam ↔ geo-loc`).

**Recommendation.** Deploy the **fine-tuned transformer** as the entity extractor, and feed entity counts per time window
into the trend-detection pipeline in place of hashtags. Next steps that will move the needle most: a Twitter-specific
cased model (BERTweet / Twitter-RoBERTa), more labelled data for rare types (active learning), and a gazetteer /
entity-linking layer for new names.
