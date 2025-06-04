import re
import tiktoken
# for testing import libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def clean_markdown(text):
    """Clean markdown text by removing Hugo shortcodes and HTML comments."""
    # Remove Hugo shortcodes like {{< >}} or {{% %}}
    text = re.sub(r'\{\{<.*?>\}\}', '', text)
    text = re.sub(r'\{\{%.*?%\}\}', '', text)
    # Remove HTML comments <!-- -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    return text.strip()

def count_tokens(text, model):
    """Count the number of tokens in a text using the specified model."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def estimate_cost(input_text, output_text, model, input_cost, output_cost):
    """Estimate the cost based on input and output tokens."""
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)
    total_cost = (input_tokens * input_cost) + (output_tokens * output_cost)
    return input_tokens, output_tokens, total_cost

# for testing
def compute_semantic_similarity(text1, text2, embedding_model):
    """Compute semantic similarity between two texts using the specified embedding model."""
    try:
        embedding1 = embedding_model.embed_query(text1)
        embedding1 = np.array(embedding1).reshape(1, -1)
        doc_embed = clean_markdown(text2)
        embedding2 = embedding_model.embed_query(doc_embed)
        embedding2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]
    except Exception as e:
        print(f"Error computing semantic similarity: {str(e)}")
        return 0.0
    
#  text_overlap => calcualtes word overlap betwen two texts as a simple grounding check.
def compute_text_overlap(text1, text2):
    """ Compute word overlap between two texts as a proxy for grounding """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    common_words = words1.intersection(words2)
    overlap = len(common_words) / max(len(words1), len(words2), 1)
    return overlap