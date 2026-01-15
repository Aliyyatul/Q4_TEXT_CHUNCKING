import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

# -----------------------------
# FIX: Download required NLTK data
# -----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Text Chunking using NLTK", layout="wide")

st.title("Text Chunking Web App (NLTK Sentence Tokenizer)")
st.write("Upload a PDF file to extract text and perform sentence-based chunking using NLTK.")

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    # -----------------------------
    # Extract text from PDF
    # -----------------------------
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + " "

    st.success("PDF text extracted successfully.")

    # -----------------------------
    # Sentence tokenization
    # -----------------------------
    sentences = sent_tokenize(full_text)

    st.subheader("Sample Extracted Sentences (Index 58 to 68)")

    if len(sentences) < 69:
        st.warning("The PDF does not contain enough sentences (less than 69).")
    else:
        sample_sentences = sentences[58:69]

        for i, s in enumerate(sample_sentences, start=58):
            st.write(f"[{i}] {s}")

        # -----------------------------
        # Sentence chunking
        # -----------------------------
        st.subheader("Sentence Chunks (NLTK)")

        for idx, chunk in enumerate(sample_sentences, start=1):
            st.markdown(f"**Chunk {idx}**")
            st.write(chunk)
            st.markdown("---")

else:
    st.info("Please upload a PDF file to begin.")
