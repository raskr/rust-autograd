#!/usr/bin/env bash

mkdir -p data/imdb
curl http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz --output "./data/imdb/aclImdb_v1.tar.gz"
tar zxvf  data/imdb/aclImdb_v1.tar.gz -C ./data/imdb
rm data/aclImdb_v1.tar.gz