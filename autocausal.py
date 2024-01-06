from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the path to your locally saved model and tokenizer
model_path = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"


# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Input text
input_text = "This is an example input."

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text with the model
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)
