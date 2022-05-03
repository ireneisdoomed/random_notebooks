#!/usr/bin/env python3

import os
import logging

from txtai.embeddings import Embeddings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_embeddings(embeddings_path):
    # Load the embeddings
    embeddings = Embeddings()
    embeddings.load(embeddings_path)

    # Return the embeddings
    print(f'{embeddings.count()} embeddings loaded')
    return embeddings

def search(query, embeddings):

    # Search the embeddings
    results = embeddings.search(query, 1)
    best_match = results[0]

    # Return the results
    print(f"Best match for {query} is {best_match['text']} ({best_match['id']}) with score {best_match['score']}")
    return best_match


if __name__ == '__main__':
    query = 'high cholesterol'
    input_embeddings = 'embeddings/embeddings_labels.tar.gz'

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    embeddings = load_embeddings(input_embeddings)
    search(query, embeddings)
