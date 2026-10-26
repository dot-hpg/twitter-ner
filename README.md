# Twitter Named Entity Recognition — BiLSTM-CRF vs BERT

Recognise **people, places, companies, products, music artists, movies, sports teams, TV shows, facilities and other
entities in tweets**, so trends can be tracked from what people actually write rather than from hashtags.

Trained and evaluated on the **WNUT-16** Twitter NER corpus (CoNLL / BIO format, 10 entity types → 21 labels), with
TensorFlow 2.15 and Hugging Face Transformers. The full write-up is in
**[Twitter_NER_GuntukuHariprasad.pdf](Twitter_NER_GuntukuHariprasad.pdf)**.

## Results

Entity-level micro F1 (seqeval, exact span + type) on the untouched WNUT-16 test set (3,849 tweets).
Models are selected on a 10% validation split, never on the test set.

| Model | Test precision | Test recall | **Test F1** |
|---|---|---|---|
| LSTM + softmax (Word2Vec) | 0.117 | 0.164 | 0.137 |
| BiLSTM + softmax (Word2Vec) | 0.137 | 0.210 | 0.166 |
| BiLSTM + CRF (Word2Vec) | 0.183 | 0.193 | 0.188 |
| BiLSTM + CRF + char-CNN (Word2Vec) | 0.213 | 0.265 | 0.236 |
| BiLSTM + CRF + char-CNN (GloVe-Twitter) | 0.479 | 0.361 | 0.412 |
| **BERT-base-uncased, fine-tuned** | – | – | **0.437** |

What mattered most: **pre-trained knowledge**. 85% of test entities never appear in the training data, so GloVe
embeddings trained on 2B tweets beat every architectural change, and fine-tuned BERT does better still.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/01_eda_bilstm_crf.ipynb` | Data loading, EDA, preprocessing, TF tokenizer, Word2Vec, BiLSTM + custom CRF, 12 experiments |
| `notebooks/02_bert_ner.ipynb` | BERT tokenization and label alignment, fine-tuning experiments, sub-token joining, `save_pretrained` |
| `src/` | Notebook sources (percent format); `tools/run_nb.py` builds and executes them |
| `serving/` | FastAPI inference service (`POST /ner`, `/health`, `/ready`) |
| `Dockerfile`, `docker-compose.yml` | Container image for the API |
| `k8s/` | Kubernetes Deployment, Service, HPA, PodDisruptionBudget, ConfigMap (kustomize) |
| `report/`, `tools/build_report.py` | Report template; renders the executed notebooks into the PDF |
| `outputs/` | Experiment results (JSON / CSV) |

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
gdown --folder https://drive.google.com/drive/folders/14IgdWzzpjp166rhNhp9UFenjUp_czaGo -O data
cd tools
python run_nb.py 01_eda_bilstm_crf      # ~1.5 h on a laptop CPU
python run_nb.py 02_bert_ner            # ~4 h on CPU, minutes on a GPU
python build_report.py                  # rebuilds the PDF from the executed notebooks
```

All runs use `SEED = 42`. Re-running notebook 01 reproduces every number exactly.

## Serve the model

The API loads the fine-tuned model from `models/bert_ner/`, which `02_bert_ner` writes with `save_pretrained`.

```bash
docker compose up --build                # http://localhost:8000/docs
curl -X POST localhost:8000/ner -H "Content-Type: application/json" \
     -d '{"texts": ["Watching Stranger Things on Netflix in London"]}'
```

Response (one item per text): the words, their BIO tags, and the grouped entities with word offsets.

## Deploy on Kubernetes

The manifests in `k8s/` run 2+ replicas with startup, readiness and liveness probes, CPU/memory requests and limits,
zero-downtime rolling updates, a PodDisruptionBudget and a HorizontalPodAutoscaler (2–6 pods at 70% CPU).

**Locally with Docker Desktop:** enable Kubernetes in *Docker Desktop → Settings → Kubernetes*, then:

```bash
docker build -t twitter-ner-api:1.0 .
kubectl apply -k k8s
kubectl -n twitter-ner rollout status deployment/ner-api
curl localhost:8080/ready
```

**On a cloud cluster (EKS / GKE / AKS):** push the image to a registry (for example
`ghcr.io/dot-hpg/twitter-ner-api:1.0`), set that name in `k8s/deployment.yaml`, change the Service to `ClusterIP`
behind an Ingress, and make sure `metrics-server` is installed so the autoscaler has CPU metrics.

## Author

**Guntuku Hariprasad**, Scaler Academy business case (NLP).
