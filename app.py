import json
import random
from logging import Logger

import nltk
import numpy as np
import csv
import logging
import requests
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sympy as sp
from flask_cors import CORS

# Download the NLTK data files if you haven't already
# nltk.download('punkt')
# nltk.download('wordnet')
nltk.download('omw-1.4')

# Load intents file
with open('inputs/intents.json') as file:
    data = json.load(file)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare the dataset
training_sentences = []
training_labels = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        # Lemmatize each word and convert it to lowercase
        training_sentences.append(' '.join([lemmatizer.lemmatize(w.lower()) for w in word_list]))
        training_labels.append(intent['tag'])
        responses[intent['tag']] = intent['responses']

# Vectorize the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences).toarray()
y = np.array(training_labels)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
# Store user feedback
feedback_data = []

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision with zero_division parameter
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)

# Calculate recall with zero_division parameter
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

# Calculate F1-score with zero_division parameter
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Evaluation metric labels
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

# Plotting the graph
plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])

# Adding title and labels
plt.title('Evaluation Metrics for Chatbot Model', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Scores', fontsize=12)

# Display the values on top of the bars
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)

# Show the plot
plt.ylim(0, 1.1)  # Set y-axis limit to ensure the bars and text fit
#plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (Weighted): {precision:.2f}")
print(f"Recall (Weighted): {recall:.2f}")
print(f"F1-Score (Weighted): {f1:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)



# Visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


def retrain_model():
    # Load updated intents.json
    with open('inputs/intents.json') as file:
        updated_data = json.load(file)

    # Update training data
    training_sentences = []
    training_labels = []
    for intent in updated_data['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            training_sentences.append(' '.join([lemmatizer.lemmatize(w.lower()) for w in word_list]))
            training_labels.append(intent['tag'])

    # Re-vectorize and retrain
    X = vectorizer.fit_transform(training_sentences).toarray()
    y = np.array(training_labels)
    model.fit(X, y)

    print("Model retrained successfully!")


import re

import re
import sympy as sp

def is_math_expression(text):
    # This function checks if the input contains a mathematical expression
    return bool(re.search(r"[\d\s+\-*/^().]+", text.strip()))
def is_Trivia_related(text):
    """
    This function checks if the input text is likely related to trivia.
    It does so by looking for common keywords associated with news topics.
    """
    trivia_keywords = [
        r"\b(trivia)\b",

    ]
    pattern = "|".join(trivia_keywords)
    return bool(re.search(pattern, text.strip(), re.IGNORECASE))


def is_news_related(text):
    """
    This function checks if the input text is likely related to news.
    It does so by looking for common keywords associated with news topics.
    """
    news_keywords = [
        r"\b(news|headline|breaking|update|report|current events|trending)\b",
       # r"\b(world|local|international|politics|sports|economy|technology|entertainment|business|science)\b",
        #r"\b(latest|today|this week|recent|ongoing)\b"
    ]
    pattern = "|".join(news_keywords)
    return bool(re.search(pattern, text.strip(), re.IGNORECASE))

import re

def is_country_related(text):
    """
    This function checks if the input text is likely related to a country.
    It does so by looking for common keywords or country-related terms.
    """
    country_keywords = [
      # r"\b(country|nation|state|capital|population|flag|currency|language|geography|history|tourism|economy|government)\b",
       # r"\b(about|in|of|from)\b",  # Often used with country names
        # List of country names (you can add as many as needed)
        r"\b(usa|united kingdom|canada|india|brazil|france|germany|china|japan|australia|italy|spain|mexico|uk|russia|south africa|argentina|south korea|turkey|sweden|norway|netherlands|poland|switzerland|saudi arabia|nigeria|egypt|indonesia|malaysia|singapore|thailand|vietnam|philippines|peru)\b"
    ]
    pattern = "|".join(country_keywords)
    return bool(re.search(pattern, text.strip(), re.IGNORECASE))

import re

def is_crypto_related(text):
    """
    This function checks if the input text is likely related to cryptocurrency.
    It does so by looking for common keywords or cryptocurrency-related terms.
    """
    crypto_keywords = [
        r"\b(cryptocurrency|crypto|bitcoin|ethereum|coin|token|blockchain|price|market|value|usd|btc|eth)\b",
        r"\b(price|current value|value of|rate|quote)\b",  # Commonly used with cryptocurrency queries
        r"\b(buy|sell|trade|exchange)\b"  # Related to cryptocurrency trading
    ]
    pattern = "|".join(crypto_keywords)
    return bool(re.search(pattern, text.strip(), re.IGNORECASE))

import re

def is_number_trivia_related(text):
    """
    This function checks if the input text is likely related to number trivia.
    It does so by looking for common keywords or number-related terms.
    """
    number_trivia_keywords = [
        r"\b(number|trivia|fact|did you know|interesting)\b",
       # r"\b(math|year|date|history|maths)\b",  # Related to mathematical or historical trivia
      #  r"\b(facts|about)\b",  # Commonly used in number-related queries
        r"\b(\d{1,6})\b"  # Any number query (e.g., "facts about 42")
    ]
    pattern = "|".join(number_trivia_keywords)
    return bool(re.search(pattern, text.strip(), re.IGNORECASE))

def solve_math_problem(problem):
    try:
        # Check if the problem is a direct arithmetic expression (e.g., 5+5, 10/2)
        if is_math_expression(problem):
            return f"Result: {eval(problem)}"

        # Parsing and solving symbolic math problems using sympy
        x = sp.symbols('x')  # Define 'x' for symbolic math
        if "derivative" in problem or "differentiate" in problem:
            # Extract the expression to differentiate
            expression = re.search(r"derivative of (.+)", problem, re.IGNORECASE).group(1)
            derivative = sp.diff(expression, x)
            return f"Derivative: {derivative}"
        elif "integral" in problem or "integrate" in problem:
            # Extract the expression to integrate
            expression = re.search(r"integral of (.+)", problem, re.IGNORECASE).group(1)
            integral = sp.integrate(expression, x)
            return f"Integral: {integral}"
        elif "solve" in problem:
            # Extract the equation to solve
            equation = re.search(r"solve (.+)", problem, re.IGNORECASE).group(1)
            solutions = sp.solve(equation, x)
            return f"Solutions: {solutions}"
        else:
            return "Sorry, I couldn't understand the problem. Could you rephrase it?"
    except Exception as e:
        return f"An error occurred: {e}"




# Define a function to get cryptocurrency data, including prices and market data.
def get_crypto_price(coin):

    response = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=cad')
    if response.status_code == 200:
        data1 = response.json()

        logger.info(f"Response from crypto API: {data1}")
        price = data1[coin]['cad']

        return f"The current price of {coin} is ${price}."
    else:
        return "Sorry, I couldn't retrieve the cryptocurrency price."
# Define a function to get trivia and facts about numbers, including math, date, and year trivia.
def get_number_trivia(number):

    logger.info(f"Trivia number passed:{number}")
    response = requests.get(f'http://numbersapi.com/{number}')
    logger.info(f"Trivia number passed:{number}")
    if response.status_code == 200:
        return response.text
    else:
        return "Sorry, I couldn't retrieve trivia for that number."


# Define a function to get jokes of various types (programming, general, etc.)
def get_joke():
    response = requests.get('https://v2.jokeapi.dev/joke/Any')
    if response.status_code == 200:
        data = response.json()
        if data['type'] == 'single':
            return data['joke']
        else:
            return f"{data['setup']} ... {data['delivery']}"
    else:
        return "Sorry, I couldn't retrieve a joke right now."

# Define a function to get information about countries, including name, population, area, and languages.
def get_country_info(country):
    response = requests.get(f'https://restcountries.com/v3.1/name/{country}')
    if response.status_code == 200:
        data = response.json()
        country_data = data[0]
        name = country_data['name']['common']
        population = country_data['population']
        area = country_data['area']
        return f"{name} has a population of {population} and covers an area of {area} square kilometers."
    else:
        return "Sorry, I couldn't retrieve information about that country."


import requests

# Define your NewsAPI key
NEWS_API_KEY = '987ef131280043b79953736593992221'



def get_todays_news(country='us'):
    """
    Fetch today's top headlines using NewsAPI.
    :param country: The country code for news localization (default: 'us').
    :return: A string with top headlines, each on a new line.
    """


    url = f"https://newsapi.org/v2/top-headlines"
    params = {
        'country': country,
        'apiKey': NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            news_data = response.json()
            if news_data['status'] == 'ok':
                headlines = [article['title'] for article in news_data['articles']]

                # If there are headlines, format them with newlines
                if headlines:
                    formatted_headlines = '/n'.join(headlines)
                    return formatted_headlines
                else:
                    return "No news articles available."
            else:
                return f"Error from NewsAPI: {news_data.get('message', 'Unknown error')}"
        else:
            return f"Failed to fetch news. HTTP Status: {response.status_code}"
    except Exception as e:
        return "Unable to connect to NewsAPI at the moment. Please try after sometime"




# Example usage



# Define your OpenWeatherMap API key and endpoint
API_KEY = '1a7a0990786eb735437aefe0916bdbcd'
WEATHER_URL = 'http://api.openweathermap.org/data/2.5/weather'
# Define a function to get the weather
def get_weather(city):
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'  # You can use 'imperial' for Fahrenheit
    }
    response = requests.get(WEATHER_URL, params=params)
    if response.status_code == 200:
        data1 = response.json()
        temperature = data1['main']['temp']
        weather_description = data1['weather'][0]['description']
        return f"The current temperature in {city} is {temperature}°C with {weather_description}."
    else:
        return "Sorry, I couldn't retrieve the weather information."


def is_weather_related(text):
    # Convert text to lowercase
    text = text.lower()

    # Regular expression pattern for weather-related queries
    pattern = r"(weather|rain|sun|snow|cloud|forecast|temperature|humidity|storm|wind|degree)"

    # Check if the text contains any weather-related keywords or patterns
    if re.search(pattern, text):
        return True
    elif text.startswith("weather in"):
        return True
    return False

# Define a function to get a response
def get_response(text):
    # Preprocess the text
    text = text.lower()
    text_vectorized = vectorizer.transform([text]).toarray()
    # Predict the intent
    predicted_label = model.predict(text_vectorized)[0]

    logger.info(f"predicted_label for {text}:"+predicted_label)
    # Check if the intent is for mathematical problem
    if predicted_label == 'math' or is_math_expression(text):
        return solve_math_problem(text)
    elif predicted_label == 'weather' or is_weather_related(text):
        # Extract the city from the user input (simple example)
        city = text.split("in")[-1].strip()  # Assuming the format is "weather in city"
        return get_weather(city)
    elif predicted_label == 'country' or is_country_related(text):
        country = text.split("about")[-1].strip()
        return get_country_info(country)
    elif predicted_label == 'joke':
        return get_joke()
    elif predicted_label == 'trivia' or is_number_trivia_related(text):
        number = text.lower().split("trivia about ")[-1].strip()
        return get_number_trivia(number)
    elif predicted_label == 'news' or is_news_related(text):
        return get_todays_news()
    elif predicted_label == 'crypto' or is_crypto_related(text):
        coin = text.split("price of")[-1].strip()
        return get_crypto_price(coin)


    return random.choice(responses[predicted_label])

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
# Set up Flask
app = Flask(__name__)
CORS(app)
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")

    response = get_response(user_input)
    return jsonify({"response": response})

def is_math_expression(message):
    # Use regular expressions to check if the message contains math-like patterns
    # Basic pattern for arithmetic expressions (addition, subtraction, multiplication, division, etc.)
    pattern = r'^[\d+\-*/().^ ]+$'
    return bool(re.match(pattern, message))

if __name__ == "__main__":
    app.run(debug=True)
feedback_data = []  # Store feedback temporarily, or use a database for persistence


@app.route('/feedback', methods=['POST'])
def feedback():
    user_message = request.json.get("user_message")
    bot_response = request.json.get("bot_response")
    feedback = request.json.get("feedback")

    # Open the CSV file in append mode
    with open('feedback_data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writerow(["user_message", "bot_response", "feedback"])

        # Write the feedback data to the file
        writer.writerow([user_message, bot_response, feedback])

    # Return a response to acknowledge feedback receipt
    return jsonify({"message": "Feedback received."})


@app.route('/retrain', methods=['GET'])
def retrain():
    X_train = []
    y_train = []

    # Read feedback from the CSV file
    with open('feedback_data.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        # Process each row and use feedback for retraining
        for row in reader:
            user_message, bot_response, feedback = row
            if feedback == 'positive':
                X_train.append(user_message)
                y_train.append('correct')
            elif feedback == 'negative':
                X_train.append(user_message)
                y_train.append('incorrect')

    # Convert text to features
    X = vectorizer.fit_transform(X_train).toarray()
    y = np.array(y_train)

    # Retrain the model
    model.fit(X, y)

    return jsonify({"message": "Model retrained successfully!"})
