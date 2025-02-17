import streamlit as st
import torch
import pickle
import numpy as np
import torch.nn as nn

# Load the model
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=128, output_dim=None):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.relu(self.fc1(x[:, -1, :]))  # Use last hidden state
        x = self.log_softmax(self.fc2(x))
        return x

# Load word mappings
with open("word_mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

word_to_index = mappings["word_to_index"]
index_to_word = mappings["index_to_word"]
vocab_size = len(word_to_index)

# Load trained model
model = GRUModel(vocab_size=vocab_size, output_dim=vocab_size)
model.load_state_dict(torch.load("Roman_Urdu_model.pth", map_location=torch.device("cpu")))
model.eval()

# Function to generate poetry
def generate_poetry(start_word, num_words=20):
    if start_word not in word_to_index:
        return "Word not found in vocabulary!"

    words = [start_word]
    input_seq = torch.tensor([[word_to_index[start_word]]], dtype=torch.long)

    for _ in range(num_words - 1):
        with torch.no_grad():
            output = model(input_seq)
            next_word_index = torch.argmax(output, dim=1).item()
            next_word = index_to_word.get(next_word_index, "<UNK>")
            words.append(next_word)
            input_seq = torch.tensor([[next_word_index]], dtype=torch.long)

    # Format poetry (break lines every 5 words)
    formatted_poetry = "\n".join([" ".join(words[i:i+5]) for i in range(0, len(words), 5)])
    return formatted_poetry

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Intazara Yaar Main Betha Hun</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Roman Urdu Ghazal Generator</h3>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("---")
st.markdown("🔹 **Project Description:** This model generates Roman Urdu poetry using a trained GRU model. It predicts the next words in a ghazal sequence based on an input word.")
# Input boxes
col1, col2 = st.columns(2)

with col1:
    start_word = st.text_input("Enter a starting word:", value="ishq")
    num_words = st.slider("Number of words in poetry:", min_value=5, max_value=50, value=20)

with col2:
    st.markdown("### Ghazal Generated:")
    if st.button("Generate Poetry"):
        poetry = generate_poetry(start_word, num_words)
        st.text_area("", poetry, height=250)



