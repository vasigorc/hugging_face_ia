from transformers import BertTokenizer, BertModel
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

input_text = """
After a long day at work, Sarah decided to relax by taking her 
dog for a walk in the park. As they strolled along the 
tree-lined paths, Sarah's dog, Max, eagerly sniffed around, 
chasing after squirrels and birds. Sarah smiled as she watched 
Max enjoy himself, feeling grateful for the companionship and 
joy that her furry friend brought into her life."""

# get tokens for the text input, plus
# tensors like:
# tokens["input_ids"] - a tensor of shape (batch_size, sequence_length) holding the token ids for the input text
# tokens["attention_mask"] - a tensor indicating which tensors should be attended to (1 for real tokens, 0 for padding)
# pt here stands for PyTorch tensors
tokens = tokenizer(input_text, return_tensors="pt")

# .no_grad() reduces gradient caclulation, useful for reducing memory consumption
with torch.no_grad():
    # past tensors to the BERT model
    outputs = model(**tokens)

# embeddings are generated here
# returns a tensor of shape (batch_size, sequence_length, hidden_size) containing the contextual embeddings
# for each token from the last layer of BERT
last_hidden_states = outputs.last_hidden_state
print("Token embeddings:")
for token, embedding in zip(tokens["input_ids"][0], last_hidden_states[0]):
    word = tokenizer.decode(int(token))
    print(f"{word}: {embedding}")

embeddings = model.embeddings
positional_embeddings = embeddings.position_embeddings.weight
position_ids = torch.arange(tokens["input_ids"].size(1), dtype=torch.long).unsqueeze(0)
input_positional_embeddings = positional_embeddings[position_ids]

print(f"Positional embeddings shape: {input_positional_embeddings.shape}")
print("Positional embeddings shape for each token:")

for token_id, pos_embedding in zip(
    tokens["input_ids"][0], input_positional_embeddings[0]
):
    token = tokenizer.decode([token_id])
    print(f"{token}: {pos_embedding}")

# visualization using t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_tsne = tsne.fit_transform(last_hidden_states[0])

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker="o")
for i, word in enumerate(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])):
    plt.annotate(word, xy=(embeddings_tsne[i, 0], embeddings_tsne[i, 1]), fontsize=10)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of Token Embeddings")
plt.show()
plt.savefig("chapter_03/token_embeddings.png")
