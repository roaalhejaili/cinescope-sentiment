# CineScope — Sentiment Detection & Negative Tone Rewriting

**AI312 — Natural Language Processing**
Prince Muqrin University

A sentiment analysis system for movie reviews that classifies text as positive or negative using three models, then rewrites negative reviews into a constructive, professional tone using a large language model.

---

## Overview

Online reviews are often harsh in ways that make them hard to act on. This project builds a two-stage pipeline:

1. **Detect** — classify a review's sentiment using Logistic Regression, SVM, or a CNN
2. **Rewrite** — if the tone is negative, generate a constructive rephrasing that preserves the original criticism without softening it into false praise

The rewriting stage was originally rule-based (keyword → canned sentence) and was replaced with an LLM, which produces rewrites grounded in the actual content of the review.

---

## Demo

**Negative review — all three models agree, each produces its own rewrite**

![All models on a negative review](screenshots/negative-all-models.png)

**Mixed sentiment — praise is preserved, criticism is rephrased**

![Mixed sentiment review](screenshots/mixed-sentiment.png)

**Positive review — correctly identified, no rewrite triggered**

![Positive review](screenshots/positive-no-rewrite.png)

**Single-model view with confidence breakdown**

![Single model view](screenshots/single-model-confidence.png)

---

## Dataset

IMDB Movie Reviews — 50,000 labelled reviews, balanced 25,000 positive / 25,000 negative.
Split 80/20 into train and test, stratified by label: **40,000 train / 10,000 test**.

---

## Pipeline

### Preprocessing
Lowercasing → HTML tag removal → URL removal → punctuation removal → digit removal → stopword removal → lemmatization (WordNet).

### Feature representations
Three representations were compared using Logistic Regression as a fixed baseline:

| Representation | Accuracy | F1-Score |
|---|---|---|
| Bag-of-Words (10k features) | 0.8709 | 0.8714 |
| **TF-IDF (1–2 grams, 10k features)** | **0.8935** | **0.8945** |
| Word2Vec (100d) | 0.8536 | 0.8543 |

TF-IDF performed best and was used for the final classical models.

### Models

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.8935** | 0.8863 | **0.9028** | **0.8945** |
| SVM (LinearSVC) | 0.8845 | 0.8784 | 0.8926 | 0.8854 |
| CNN | 0.8762 | 0.8651 | 0.8914 | 0.8781 |

Best model by F1-Score: **Logistic Regression**, used as the default in the interactive demo.

- **Logistic Regression** — TF-IDF features, `max_iter=1000`
- **SVM** — LinearSVC on the same TF-IDF features
- **CNN** — Embedding (128d) → Conv1D (128 filters, kernel size 5) → GlobalMaxPooling → Dense; sequences padded to 200 tokens, 20k-word vocabulary, 5 epochs

The CNN reached 99.5% training accuracy by epoch 4 while validation accuracy stalled near 87.9%, indicating overfitting — the classical models generalise better on this dataset at this scale.

### Tone rewriting

Negative reviews are passed to the Gemini API with a constrained system prompt instructing the model to keep the specific criticisms and concrete details, avoid turning the review positive, avoid inventing content, and preserve approximate length.

The implementation rotates across several Gemini Flash models, since each carries its own daily free-tier quota, and applies exponential backoff on transient server errors.

---

## Results

### Rule-based vs. LLM rewriting

The original rule-based system matched keywords against a fixed list and returned a canned sentence:

| Input | Rule-based output |
|---|---|
| "This movie was terrible and a complete waste of time." | "The experience was not fully satisfying and has significant room for improvement." |
| "The plot was so boring and predictable." | "The content was not very engaging and could benefit from more compelling elements." |
| "Absolutely loved this film!" | "The experience did not fully meet expectations, but there is room for improvement." |

Two failure modes are visible here: the output never references the film, and the third example shows the rules firing on a positive review because they run independently of the classifier.

By contrast, the LLM rewrites stay grounded in the review itself:

| Input | LLM rewrite |
|---|---|
| "wasted my time for nothing worst movie" | "The film failed to engage my interest, and I found the overall experience unrewarding and unsatisfying." |
| "I went in expecting a decent thriller and got two hours of nothing" | "The film struggled to deliver the engaging thriller experience I had anticipated, resulting in a two-hour runtime that felt uneventful and lacked momentum." |
| "The costumes and score are genuinely gorgeous, but the script is boring and the ending felt rushed and unearned." | "While the costumes and score are genuinely gorgeous, the script struggles with pacing issues that render it somewhat unengaging, and the resolution could benefit from more development to feel fully earned." |

The LLM rewrites keep the specific content. For the input *"The costumes and score are genuinely gorgeous, but the script is boring and the ending felt rushed and unearned"*, the system returns a rephrasing that retains the praise for costumes and score while restating the pacing and ending criticisms constructively.

### Observations

- On mixed-sentiment reviews, the LLM preserves positive elements rather than flattening the whole review into criticism.
- TF-IDF models struggle with sarcasm, where positive vocabulary carries negative intent — a known limitation of bag-of-words representations.
- Model confidence varies meaningfully across inputs: unambiguous reviews score above 99%, while mixed-sentiment inputs drop closer to 54%, which is a useful signal for when a rewrite should be trusted.

---

## Setup

```bash
pip install scikit-learn tensorflow nltk pandas numpy ipywidgets gensim google-genai
```

1. Place `IMDB Dataset.csv` in the project directory
2. Get a free API key from [Google AI Studio](https://aistudio.google.com)
3. In Colab, add it under the key icon as a secret named `GOOGLE_API_KEY`. Running locally, the notebook falls back to a `getpass` prompt
4. Run the notebook top to bottom

**Running without an API key:** the sentiment classifiers train and run normally without one — only the rewriting stage needs it. Example outputs from the rewriting stage are shown in the Results section above.

**Free-tier quota:** Gemini's free tier allows 20 requests per day per model. The notebook rotates through several Flash models automatically when one is exhausted, and retries with exponential backoff on transient server errors.

---

## Project structure

```
.
├── AI312_NLP_Project.ipynb    # main notebook
├── screenshots/               # demo images
└── README.md
```

---

## Limitations & future work

- Sarcasm and negation are handled poorly by the TF-IDF models; a transformer classifier such as BERT would likely close this gap
- The CNN overfits within 4 epochs — dropout, early stopping, or pretrained embeddings would help
- Rewriting quality is currently assessed qualitatively; adding a tone-flip rate (re-classifying rewrites with the project's own model) and a semantic similarity score would make the evaluation quantitative
- Free-tier API quotas cap the size of any automated rewriting evaluation
