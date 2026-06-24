# Preprocessing

This repository uses shared preprocessing so experiment and evaluation tasks can compare
models under explicit, reproducible conditions.

## Runtime Prerequisites

Shared preprocessing expects the following runtime resources:

- English lemmatization uses NLTK `WordNetLemmatizer`
- the `nltk` package is installed by Poetry
- the NLTK `wordnet` corpus is downloaded by the project setup script
- Japanese tokenization uses the MeCab Python bindings plus a dictionary such as
  `mecab-ipadic-neologd`, `unidic`, or `unidic-lite`
- the MeCab bindings and Japanese dictionaries are runtime prerequisites, not packages
  installed from this repository's `pyproject.toml`

Install the English lemmatization corpus with:

```bash
poetry run setup-nltk
```

## Shared Configuration Axes

Config may declare the following shared preprocessing axes under `dataset` and/or
`preprocess`:

- `language`
- `delimiter`
- `segmenter`
- `tokenizer`
- `text_column`
- `target_column`
- `ja_replace_num`
- `ja_stopwords_path`
- `ja_dicdir`
- `ja_require_unidic`

Default policy:

- `segmenter=delimiter`
- `tokenizer=default`
- `default` resolves to `simple` for English and `mecab` for Japanese
- stopword removal is not part of the shared preprocessing contract
- POS-based token filtering is not part of the shared preprocessing contract
- stemming is not part of the shared preprocessing contract
- English tokens are lemmatized with WordNet
- Japanese tokens use MeCab surface segmentation with lemma or base-form normalization
- numeric tokens are normalized to `<NUM>` for both English and Japanese
- pure punctuation or symbol tokens are removed for both English and Japanese
- `ja_stopwords_path` is still parsed and persisted for compatibility, but the shared
  tokenizer does not currently remove Japanese stopwords
- when `ja_dicdir` is omitted, the runtime prefers `mecab-ipadic-neologd`, then
  `unidic` or `unidic-lite`; with `ja_require_unidic=false`, MeCab's default dictionary
  can still be used when available

## Tokenization Policy

The shared tokenizer keeps one cross-language policy:

- split text into sentences first
- tokenize each sentence into lexical tokens
- avoid manually removing stopwords or function words
- avoid POS-based content-word filtering
- normalize inflectional variants with lemmatization or base-form conversion
- leave corpus-level vocabulary filtering to downstream dictionary construction

This means the repository keeps tokens such as English auxiliaries or Japanese particles
when they are emitted by the tokenizer. The shared layer does not try to decide whether
a token is semantically useful for topic modeling. That decision is deferred to later
statistical filtering such as document-frequency thresholds.

### English

- tokenization extracts alphabetic and numeric tokens
- alphabetic tokens are lowercased and lemmatized with WordNet
- numeric tokens are replaced with `<NUM>`
- punctuation-only tokens are dropped

### Japanese

- tokenization uses MeCab with the configured dictionary
- tokens are kept without POS-based exclusion
- lemma or base-form values are preferred when available
- numeric tokens are replaced with `<NUM>`
- punctuation-only tokens are dropped

## Shared Outputs

Shared preprocessing may persist document views such as:

- `sentences_raw`
- `sentences_tokenized`
- `sentences_joined`
- `document_tokens`

These views are reused across models and analysis tasks to keep comparisons aligned.

## Model Usage

- vMF Sentence LDA (vSLDA) consumes shared raw sentence strings
- CTM consumes shared document tokens plus raw or contextual text inputs
- BleiLDA, Gaussian LDA, and ETM consume shared document tokens
- SenClu and Sentence Gaussian LDA consume shared raw sentence strings

ETM is implemented as a document-level BoW baseline. It uses pre-fitted word
embeddings by default through the same `word2vec` and optional
`wikientvec_cache_dir` parameters as Gaussian LDA and MvTM, and it does not emit
sentence-topic artifacts.

The intent is that lexical tokenization and sentence segmentation happen once in
repo-owned code rather than being redefined inside each model path.

## Ownership Boundary

Repo-owned shared logic covers:

- CSV loading and target filtering
- category-to-target resolution
- shared sentence splitting and tokenization
- canonical preprocessed document structures
- preprocessing metadata recorded in artifacts

Model-specific code should only add the transformations that are genuinely
model-specific after the shared preprocessing stage.
