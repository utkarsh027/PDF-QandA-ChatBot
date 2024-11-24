# PDF-QandA-ChatBot with streamlit

Here’s a detailed README.md for your Q&A app using RAG, OpenAI embeddings, PDF retrieval, and Streamlit:

RAG-Based Q&A App with Streamlit
This project implements a Retrieval-Augmented Generation (RAG) approach for creating a Q&A chatbot. The app uses Hugging Face embeddings to process and retrieve information from any PDF document and leverages Streamlit to provide an interactive user interface.

The system enables users to ask questions about the content of the PDF and receive accurate responses based on the document.

# FEATURES:

PDF Document Retrieval: Uses any PDF as the primary knowledge source.
HuggingFace Embeddings: Embeds PDF content for effective retrieval of relevant context.
RAG Architecture: Combines document retrieval and language generation for precise answers.
Interactive Interface: Built with Streamlit for a smooth user experience.
Customizable API Keys: Reads HuggingFace and Groq API keys securely using environment variables.
# Prerequisites
Python 3.10 or higher
HuggingFace anf Groq API keys
Required Python libraries (see requirements.txt)

# Follow this

# Step 1: Clone the Repository
# Step 2: Create a Virtual Environment
# Step 3: Install Dependencies
pip install -r requirements.txt
# Step 4: Configure API Keys
Hugging Face
Groq

# Running the App
python -m streamlit run app.py
