import argparse
from vectordb.vector_db import VectorDB
from embeddings.sgpt_embedder import SGPTEmbedder
from misc.file_fns import load_pdf, read_yaml
from model_wrappers.gpt_wrapper import GPTWrapper


if __name__ == '__main__':
    embedder = SGPTEmbedder()
    gpt_wrapper = GPTWrapper()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_filepath', type=str)
    parser.add_argument('--prompt_type', type=str, default='summarization_excerpt')

    args = parser.parse_args()
    pdf_filepath = args.pdf_filepath
    prompt_type = args.prompt_type
    prompt_dict = read_yaml('ref/prompts.yml')
    excerpt_str = prompt_dict.get(prompt_type, 'summarization')

    print('LOADING PDF')
    pdf_content_list = load_pdf(pdf_filepath, split_char='. ')
    embedding_dims = embedder.get_sentence_embedding_dimension()
    print('EMBEDDING TEXT')
    embedded_text = embedder.encode(pdf_content_list)

    print('ADDING EMBEDDINGS TO DATABASE')
    vector_index = VectorDB(n_dims=embedding_dims)
    vector_index.add_vectors(vectors=embedded_text, metadata=pdf_content_list)

    while True:
        print('Enter a topic from your pdf\n?:')
        input_context = input()
        test_str_embedded = embedder.encode(input_context)
        output = vector_index.get_neighbors(vector=test_str_embedded, return_metadata=True, k=10)
        context_to_insert = '\n'.join([x['metadata'] for x in output])
        input_kwargs = {'context': context_to_insert, 'topic': input_context}
        if prompt_type in ['context']:
            print('Enter your question about the previously entered topic\n?:')
            input_question = input()
            input_kwargs.update({'question': input_question})
        elif prompt_type in ['summarization_excerpt']:
            input_kwargs.update({'question': input_context})
        prompt_formatted = excerpt_str.format(**input_kwargs)
        response_msg = gpt_wrapper.send_message(prompt_formatted)
        print(response_msg, end='\n==============================\n')

