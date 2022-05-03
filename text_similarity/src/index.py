#!/usr/bin/env python3

import os
import logging

import pandas as pd
from txtai.embeddings import Embeddings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main(features: str, output_path: str):
    # Create the embeddings indicating we want to store the content of the text, not only the vectors
    embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True, "objects": True})

    # Add the sentences
    embeddings.index([(uid, text, None) for uid, text in features])

    # Save the embeddings
    embeddings.save(output_path)
    print(f'{embeddings.count()} embeddings saved to {output_path}')


if __name__ == '__main__':
    input_data = 'data/documents.json'
    output_embeddings = 'embeddings/embeddings_synonyms.tar.gz'

    # Load the data
    df = pd.read_json(input_data)
    features_synonyms = df[['id', 'text']].drop_duplicates().to_records(index=False).tolist()

    main(features_synonyms, output_embeddings)
