import gradio as gr

from src.search import load_embeddings


def search(query):

    # Search the embeddings
    results = embeddings.search(query, 1)
    best_match = results[0]

    # Return the results
    print(f"Best match for {query} is {best_match['text']} ({best_match['id']}) with score {best_match['score']}")
    return best_match

if __name__ == "__main__":

    embeddings_path = 'embeddings/embeddings_synonyms.tar.gz'
    embeddings = load_embeddings(embeddings_path)
    demo = gr.Interface(fn=search, inputs='text', outputs='text')

    demo.launch(share=True)