import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load data from CSV file
data = pd.read_csv('C:/Users/Jacqy/Documents/bot/bot.csv')

# Display the columns to verify their names and contents
st.write("Columns in the dataset:")
st.write(data.columns)
st.write("\nFirst 5 rows of the dataset:")
st.write(data.head())

def clean_text(text):
    if isinstance(text, str):
        text = text.lower().strip()  # Convert to lowercase and remove leading/trailing whitespace
    return text

# Adjust column names if necessary (replace with actual column names from your dataset)
data.rename(columns={'Question': 'question', 'Answer': 'context'}, inplace=True)

# Clean data
data['question'] = data['question'].apply(clean_text)
data['context'] = data['context'].apply(clean_text)

# Handle NaN values in 'context' column (if any)
data['context'].fillna("", inplace=True)

# Initialize the question-answering pipeline
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Define a function to get the answer for each question-context pair
def get_answer(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        print(f"Error processing question: {question} with context: {context}. Error: {e}")
        return None

# Apply the function to your data to get the predicted answers
data['predicted_answer'] = data.apply(lambda row: get_answer(row['question'], row['context']), axis=1)

# Display the data with the predicted answers
st.write("\nData with predicted answers:")
st.write(data[['question', 'context', 'predicted_answer']].head())

# Create a dictionary from the dataset for fast lookup
qa_dict = data.set_index('question')['predicted_answer'].to_dict()

# Function to clean and preprocess user input
def clean_user_input(text):
    text = text.strip().lower()  # Convert to lowercase and remove leading/trailing whitespace
    return text

# Function to handle user queries
def chatbot(user_input, qa_dict):
    user_input = clean_user_input(user_input)
    
    if user_input in qa_dict:
        return [qa_dict[user_input]]
    else:
        return [f"Sorry, I don't know the answer to '{user_input}'."]

# Streamlit app
def main():
    st.title("Mental Health Chatbot")
    st.write("Welcome to the Mental Health Chatbot! Type your question below.")

    user_input = st.text_input("You: ")

    if st.button("Ask"):
        answers = chatbot(user_input, qa_dict)
        for answer in answers:
            st.write(f"Chatbot: {answer}")

if __name__ == "__main__":
    main()
