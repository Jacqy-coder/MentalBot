import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load and prepare data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Clean data function
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower().strip()
        return text

    data.rename(columns={'Question': 'question', 'Answer': 'context'}, inplace=True)
    data['question'] = data['question'].apply(clean_text)
    data['context'] = data['context'].apply(clean_text)
    data['context'].fillna("", inplace=True)

    # Create a dictionary for flashcards
    qa_dict = data.set_index('question')['context'].to_dict()
    
    return data, qa_dict

# Load data
data, qa_dict = load_data('C:/Users/Jacqy/Documents/bot/bot.csv')

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

# Apply the function to get the predicted answers
data['predicted_answer'] = data.apply(lambda row: get_answer(row['question'], row['context']), axis=1)

# Create a dictionary for chatbot
chatbot_dict = data.set_index('question')['predicted_answer'].to_dict()

# Function to clean and preprocess user input
def clean_user_input(text):
    return text.strip().lower()

# Function to handle user queries for the chatbot
def chatbot(user_input, chatbot_dict):
    user_input = clean_user_input(user_input)
    if user_input in chatbot_dict:
        return [chatbot_dict[user_input]]
    else:
        return [f"Sorry, I don't know the answer to '{user_input}'"]

# Streamlit app
def main():
    st.title("Mental Health Support App")
    st.write("Select a question from the dropdown to view the answer or type your question below to chat with the bot.")

    # Flashcard Interface
    st.subheader("Flashcards")
    
    # Get the list of questions
    questions = list(qa_dict.keys())

    # Layout with two columns for the flashcard interface
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_question = st.selectbox("Choose a question:", questions)

    with col2:
        if selected_question:
            question_text = selected_question.capitalize()
            answer_text = qa_dict[selected_question].capitalize()
            st.markdown(
                f"""
                <div class="flip-card">
                  <div class="flip-card-inner">
                    <div class="flip-card-front">
                      <h2>Question:</h2>
                      <p>{question_text}</p>
                    </div>
                    <div class="flip-card-back">
                      <h2>Answer:</h2>
                      <p>{answer_text}</p>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Custom CSS for the flipping card
    st.markdown(
        """
        <style>
        .flip-card {
          background-color: transparent;
          width: 600px;
          height: 400px;
          perspective: 1000px;
          margin-top: 50px;
        }
        .flip-card-inner {
          position: relative;
          width: 100%;
          height: 100%;
          text-align: center;
          transition: transform 0.6s;
          transform-style: preserve-3d;
          box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .flip-card:hover .flip-card-inner {
          transform: rotateY(180deg);
        }
        .flip-card-front, .flip-card-back {
          position: absolute;
          width: 100%;
          height: 100%;
          backface-visibility: hidden;
          border-radius: 10px;
          padding: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          border: 2px solid #d3d3d3;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .flip-card-front {
          background-color: #f9f9f9;
        }
        .flip-card-back {
          background-color: #333;
          color: white;
          transform: rotateY(180deg);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Chatbot Interface
    st.subheader("Chatbot as Per the Questions Listed")
    user_input = st.text_input("You: ")

    if st.button("Ask"):
        answers = chatbot(user_input, chatbot_dict)
        for answer in answers:
            st.write(f"Chatbot: {answer}")

if __name__ == "__main__":
    main()

