# Deep-Learning-Based-AI-Answer-Evaluator-with-Chatbot

An end-to-end NLP-driven solution that evaluates the accuracy of AI-generated answers by comparing them with expert-verified responses, using a combination of deep learning and traditional NLP techniques. 


## Project Summary

This project implements a Streamlit-based chatbot integrated with the ChatGPT API, coupled with an advanced answer evaluation engine. The engine uses BERT embeddings, LSTM models, and Siamese Networks to compute semantic similarity, along with standard metrics like Cosine Similarity, BLEU, and ROUGE scores. The system provides real-time feedback on how closely an AI-generated answer matches a domain-expert response.


## Key Objectives

- Develop an intelligent system to evaluate chatbot-generated responses.
- Leverage deep learning models (LSTM, BERT, Siamese Networks) for semantic comparison.
- Provide automated scoring and feedback using a hybrid of deep learning and rule-based metrics.
- Create a user-friendly chatbot interface using Streamlit.



## Technologies & Tools

| Category                 | Tools Used                                           |
|--------------------------|------------------------------------------------------|
| **Programming Language** | Python                                               |
| **Web Interface**        | Streamlit                                            |
| **APIs**                 | OpenAI ChatGPT API                                   |
| **Deep Learning**        | TensorFlow, Keras, LSTM, Siamese Networks            |
| **NLP Libraries**        | HuggingFace Transformers, NLTK, spaCy                |
| **Evaluation Metrics**   | Cosine Similarity, BLEU Score, ROUGE Score           |



## Core Features

**Chatbot Interface**  
  Real-time chatbot built with Streamlit and OpenAI's GPT API to generate answers for user-input questions.

**Answer Evaluation Engine**  
  Compares the generated answer with an expert-labeled answer using:
  
  - **Cosine Similarity** (Vector Space Similarity)
  - **BLEU Score** (N-gram Precision)
  - **ROUGE Score** (Recall-based)
  - **LSTM and Siamese Networks** (Deep Semantic Matching)

**Files Included**
  
  - Datasets                    
  - Wikipedia - QA-train.xlsx	     → Training dataset with question–answer pairs
  - Wikipedia - QA-validation.xlsx → Validation set used for tuning and intermediate evaluation
  - Wikipedia - QA-test.xlsx	     → Test set used for final model performance analysis.
  - Deep Learning - ChatBot.py → Deep learning model definitions (LSTM, Siamese Network, and hybrid logic)
  - app.py                     → Streamlit frontend for user interaction with chatbot and evaluation UI
  - Output Result              → ChatBot Image

**Automated Feedback**  
  Delivers interpretive feedback in real-time, based on similarity scores (range 0 to 1).

**Hybrid Architecture**  
  Combines **BERT embeddings** with **LSTM** networks for enhanced contextual understanding.




