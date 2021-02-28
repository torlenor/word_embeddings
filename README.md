# Word embeddings

[![Build status](https://git.abyle.org/torlenor/word_embeddings/badges/master/pipeline.svg)](https://git.abyle.org/torlenor/word_embeddings/commits/master)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

## Description

A collection of visualizations for exploring word embeddings in Word2Vec models. The code is dirty, the project not well set up, it may break, your computer may burn, but when it works it looks pretty.

## How to build it

### Requirements

- Python3
- Poetry

### Building

Clone the sources from the repository and compile it with

```bash
poetry install --no-dev
```

or if you need to development dependencies (for example to train models)

```bash
poetry install
```

## How to run it

### Directly

This project uses [streamlit](https://www.streamlit.io/). To run it type

```bash
poetry run streamlit run main.py
```

The models have to be in a subfolder called `models` besides main.py. So it should run out of the box with the example models when you clone this repository.

### With Docker

Build it with

```bash
docker build -t word-embeddings:latest .
```

then run it with

```bash
docker run -it -v `pwd`/models:/code/models word-embeddings:latest
```

where the volume that is mounted contains the Word2Vec models.

## Supported model formats

Note: You can add a description file by creating a text file with the same name as the model but with extension `.txt`.

### word2vec binary

The models must have the extension `.bin` and placed in the models directory.

### Gensim model

The models must have the extension `.model` and placed in the models directory.

### Gensim KeyedVectors

The models must have the extension `.kv` and placed in the models directory.

## Included models

- **word2vec_abc_news**: A model trained from the https://www.kaggle.com/therohk/million-headlines dataset. The dataset contains over a million news headlines published over a period of eighteen years from the Australian Broadcasting Corporation (ABC).

- **word2vec_bigram_abc_news**: A model trained from the https://www.kaggle.com/therohk/million-headlines dataset. The dataset contains over a million news headlines published over a period of eighteen years from the Australian Broadcasting Corporation (ABC). It is similar to the word2vec_abc_news model, but it contains bigrams (try searching for new_york).


## Thanks to

- https://github.com/marcellusruben/Word_Embedding_Visualization for PCA and t-SNA visualization and making me aware of Streamlit.
- [streamlit](https://www.streamlit.io/) for creating an awesome piece of software.
