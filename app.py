import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("Sentence Similarit Detection (मराठी वाक्य साम्य तपासणी)")
model = SentenceTransformer("sangambhamare/MarathiSentenceSimilarity")

sentence1 = st.text_input("पहिलं वाक्य:")
sentence2 = st.text_input("दुसरं वाक्य:")

if sentence1 and sentence2:
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
    similarity_score = float(util.pytorch_cos_sim(embeddings1, embeddings2)[0][0])
    st.success(f"साम्य स्कोअर: {similarity_score:.3f}")
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("मराठी वाक्य साम्य तपासणी")
model = SentenceTransformer("sangambhamare/MarathiSentenceSimilarity")

sentence1 = st.text_input("पहिलं वाक्य:")
sentence2 = st.text_input("दुसरं वाक्य:")

if sentence1 and sentence2:
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
    similarity_score = float(util.pytorch_cos_sim(embeddings1, embeddings2)[0][0])
    st.success(f"Similarity Score (साम्य स्कोअर): {similarity_score:.3f}")
