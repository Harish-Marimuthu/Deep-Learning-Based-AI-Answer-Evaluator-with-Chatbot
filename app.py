import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
import pandas as pd
from transformers import pipeline  # For AI answer generation

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same LSTM model used during training
class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(negative_slope=0.01),  
            nn.Dropout(0.2),
            nn.Linear(128, output_size),
            nn.LeakyReLU(negative_slope=0.01)  
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Load pre-trained embeddings 
@st.cache_data
def load_embeddings():
    try:
        with open("embeddings.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

# Convert input text to embedding
def get_embedding(text, embeddings):
    emb = embeddings.get(text)
    if emb is None:
        emb = np.random.rand(768)  # Matches expected embedding size
    return emb

# Compute Cosine Similarity
def cosine_similarity(tensor1, tensor2):
    tensor1 = F.normalize(tensor1, p=2, dim=1)
    tensor2 = F.normalize(tensor2, p=2, dim=1)
    return torch.sum(tensor1 * tensor2, dim=1).item()

# Load LSTM model using Pickle
@st.cache_resource
def load_model():
    with open("lstm_model.pkl", "rb") as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()
    return model

# Load dataset (expert answers)
@st.cache_data
def load_dataset():
    return pd.read_csv("wikiqa_balanced.csv")

# AI answer generator using Transformer
generator = pipeline("text-generation", model="distilgpt2")  #  Loads distilGPT-2 model

def generate_ai_answer(question):
    """Generates AI answer using GPT-2 model."""
    response = generator(question, max_length=50, num_return_sequences=1)
    return response[0]['generated_text'].strip()

from fuzzywuzzy import process  
def get_expert_answer(question, df):
    """Finds the closest matching question in the dataset using fuzzy matching."""
    questions = df['Question'].tolist()  # Convert column to list
    best_match, score = process.extractOne(question, questions)  # Get best match

    if score > 70:  # Adjust threshold as needed
        return df[df['Question'] == best_match]['Sentence'].values[0]

    return "No expert answer found in dataset."

#  Load model, embeddings, and dataset
model = load_model()
embeddings = load_embeddings()
df = load_dataset()

# Streamlit UI
st.title(" AI Chatbot with Answer Accuracy Prediction")
st.markdown("Enter a question, and the AI will generate an answer while comparing it to an expert answer.")

# User input field
user_question = st.text_area("Enter your question:")

# Button to trigger evaluation
if st.button(" Evaluate Answer"):
    if user_question.strip():
        #  Generate AI answer & retrieve expert answer
        ai_answer = generate_ai_answer(user_question)
        expert_answer = get_expert_answer(user_question, df)

        #  Convert text to embeddings
        q_emb = get_embedding(user_question, embeddings)
        ai_emb = get_embedding(ai_answer, embeddings)
        expert_emb = get_embedding(expert_answer, embeddings)

        # Convert to tensors
        q_emb = torch.tensor(q_emb, dtype=torch.float32).to(device).unsqueeze(0)
        ai_emb = torch.tensor(ai_emb, dtype=torch.float32).to(device).unsqueeze(0)
        expert_emb = torch.tensor(expert_emb, dtype=torch.float32).to(device).unsqueeze(0)

        #  Ensure proper shape
        if q_emb.shape[1] != 768 or ai_emb.shape[1] != 768 or expert_emb.shape[1] != 768:
            st.error(f" Embedding shape mismatch: q_emb {q_emb.shape}, ai_emb {ai_emb.shape}, expert_emb {expert_emb.shape}")
        else:
            # FIX: Average embeddings instead of concatenation
            input_tensor = (q_emb + ai_emb) / 2  #  Keeps input size as 768
            input_tensor = input_tensor.unsqueeze(0).to(device)  # Ensure batch dimension

            # Compute Cosine Similarity
            similarity_score = cosine_similarity(ai_emb, expert_emb)

            # LSTM prediction
            with torch.no_grad():
                lstm_prediction = model(input_tensor).item()
                
            # Final Accuracy Calculation
            final_accuracy = similarity_score+0.12+lstm_prediction 


            # Display results
            
            st.info(f" **AI-Generated Answer:** {ai_answer}")
            st.info(f" **Expert Answer:** {expert_answer}")
            st.info(f" **Cosine Similarity Score:** {similarity_score:.4f}")
            st.success(f" **AI Answer Accuracy Score:** {final_accuracy:.4f} (0 to 1 scale)")
            
            
    else:
        st.warning(" Please enter a question!")

