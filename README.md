# Mental Health Chatbot

## Overview

This repository contains a Streamlit-based chatbot designed to provide mental health support. The chatbot utilizes Transformer models for question-answering, specifically leveraging the `deepset/roberta-base-squad2` model.

## Features

- **Question-Answering**: Uses a pre-trained Transformer model to answer questions related to mental health.
- **User Interaction**: Allows users to input questions via a web interface and receive responses from the chatbot.
- **Data Integration**: Loads question-answer pairs from a CSV file (`bot.csv`) for training and validation.

## Setup Instructions

### Environment Setup:

1. Create and activate a Python environment (e.g., using Anaconda).

2. Install dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt

   ```q

   

## Running the Application:

Execute the Streamlit app:

```bash
streamlit run mental.py

```q    

Open the provided URL in a web browser to interact with the chatbot.

## Main Components

### 1. mental.py

**Data Loading and Preprocessing:**
- Loads data from `bot.csv`, cleans text, and prepares it for training the question-answering model.

**Question-Answering Pipeline:**
- Initializes a Transformer model (`roberta-base`) and a pipeline for question-answering using the Hugging Face transformers library.

**Streamlit Integration:**
- Implements a web-based interface using Streamlit for user interaction.
- Displays prompts for user input and shows chatbot responses.

**Limitation:**
- Currently, the chatbot can only respond to questions that exist in the `bot.csv` dataset.
- New questions not present in the dataset will not yield meaningful responses.

### 2. bot.csv

**Dataset Format:**
- Contains columns for questions and corresponding answers or context related to mental health.

**Usage:**
- Used for training the chatbot and providing predefined responses based on existing data.

## Limitation and Solution

### Limitation

The chatbot is limited to answering questions that are already present in the `bot.csv` dataset. Any new or unseen questions submitted by users will not receive meaningful responses.

### Solution

To improve the chatbot's responsiveness and handle new questions:

- **Continuous Learning:**
  - Implement a mechanism to collect new questions asked by users and periodically update the dataset (`bot.csv`) with new question-answer pairs.

- **Active Learning:**
  - Integrate user feedback mechanisms to refine the model and improve its ability to handle a broader range of questions over time.

