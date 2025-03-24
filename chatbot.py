import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issues for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define chatbot intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["hi", "hello", "hey", "how are you", "what's up"],
        "responses": ["Hello!", "Hey!", "I'm fine, thanks!", "Nothing much, what about you?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["bye", "see you later", "take care"],
        "responses": ["Goodbye!", "See you later!", "Take care!"]
    },
    {
        "tag": "thanks",
        "patterns": ["thank you", "thanks a lot", "I appreciate it"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do?", "Who are you?", "What are you?", "What is your purpose?"],
        "responses": ["I am a chatbot created to assist you.", "My purpose is to help answer your questions."]
    },
    {
        "tag": "help",
        "patterns": ["help", "I need your help", "Can you help me?", "What should I do?"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you?", "What's your age?"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like?", "How's the weather today?"],
        "responses": ["I'm sorry, I cannot provide real-time weather updates.", "Check a weather app for accurate info."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget?", "What's a good budgeting strategy?", "How do I create a budget?"],
        "responses": [
            "To create a budget, track your income and expenses, then allocate funds wisely.",
            "A good budgeting strategy is the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
            "Track spending, set financial goals, and stick to a plan to maintain a budget."
        ]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score?", "How do I check my credit score?", "How can I improve my credit score?"],
        "responses": [
            "A credit score represents your creditworthiness. Higher scores mean better loan approvals.",
            "You can check your credit score on free websites like Credit Karma or through your bank."
        ]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:  # Fixed incorrect loop reference
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
X = vectorizer.fit_transform(patterns)
y = tags
clf.fit(X, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't understand that."

# Streamlit app
def main():
    st.title("Chatbot")
    st.write("Welcome to the chatbot! Type a message and press Enter to start the conversation.")

    user_input = st.text_input("You:", key="user_input")
    if user_input:
        response = chatbot(user_input)
        st.write(f"Chatbot: {response}")

        if response.lower() in ['goodbye', "bye"]:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
