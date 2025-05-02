import re
import tiktoken

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
