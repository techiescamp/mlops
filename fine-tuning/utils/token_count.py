import tiktoken

# encoding = tiktoken.encoding_for_model("gpt-4.1")
encoding = tiktoken.get_encoding("cl100k_base")
print(encoding)

def num_tokens_from_message(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if not isinstance(value, str):
                value = str(value)
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message.get("role") == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, label):
    print(f"\nToken Distribution for {label}")
    print(f"\nMin / Max: {min(values)} / {max(values)}")

def calculate_cost(token_list, cost_per_million):
    total_tokens = sum(token_list)
    cost = (total_tokens / 1000000) * cost_per_million
    return total_tokens, cost
