import gradio as gr

from src.search import load_embeddings


def search(query):

    # Search the embeddings
    best_match_label = embeddings_label.search(query, 1)[0]
    best_match_synonym = embeddings_synonym.search(query, 1)[0]

    # Return the results
    return best_match_label, best_match_synonym

if __name__ == "__main__":

    # Load both models
    embeddings_label_path = 'embeddings/embeddings_labels.tar.gz'
    embeddings_synonym_path = 'embeddings/embeddings_synonyms.tar.gz'
    embeddings_label = load_embeddings(embeddings_label_path)
    embeddings_synonym = load_embeddings(embeddings_synonym_path)

    demo = gr.Interface(
        fn=search,
        inputs='text',
        outputs=['text', 'text'],
        title='Embedding based disease mapping',
        description="""
        Get the most similar disease in EFO by looking at the vector similarity. Proof of concept.
        Output 1: model only based on EFO ID/label pairs.
        Output 2: model based on EFO ID/labels and exact synonyms pairs.""",
        examples=['breast carcinoma', 'high cholesterol', 'ochoa syndrome', 'C-C motif chemokine measurement']
    )

    demo.launch(share=True)