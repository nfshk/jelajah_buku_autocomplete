from flask import Flask, render_template, request, jsonify, redirect
import json
import math
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


app = Flask(__name__)

# Load book metadata
with open('book_metadata.json') as f:
    book_metadata = json.load(f)

# ------------------- N-gram Autocomplete  ------------------- #
# Load the N-gram model
with open('ngram_model.json') as f:
    ngram_model = json.load(f)

# Fungsi untuk N-gram prediction
def predict_next_words(prefix, n=10):
    predictions = []
    words = prefix.split()
    
    if len(words) > 0:
        last_word = " ".join(words[-1:])
        if last_word in ngram_model:
            next_words_counter = ngram_model[last_word]
            total_count = sum(next_words_counter.values())
            sorted_next_words = sorted(next_words_counter.items(), key=lambda x: x[1] / total_count, reverse=True)
            for next_word, count in sorted_next_words[:n]:
                probability = count / total_count
                predictions.append((prefix + " " + next_word, probability))
    
    return predictions

# Fungsi perhitungan Perplexity
def calculate_perplexity(probability):
    if probability > 0:
        return math.exp(-math.log(probability))
    else:
        return float('inf')

# Fungsi pencarian buku berdasarkan N-gram
def search_books_by_ngram(keywords, max_results=10):
    predictions = predict_next_words(keywords)
    results = []

    for prediction, prob in predictions:
        for full_title in book_metadata.keys():
            if prediction in full_title:
                perplexity = calculate_perplexity(prob)
                results.append((full_title, prob, perplexity))

    sorted_results = sorted(results, key=lambda x: x[2])[:max_results]

    book_results = []
    for title, prob, perplexity in sorted_results:
        if title in book_metadata:
            book_data = book_metadata[title]
            book_data['probability'] = prob
            book_data['perplexity'] = perplexity
            book_results.append(book_data)
    
    return book_results

# Route to render the N-gram autocomplete page
@app.route('/ngram')
def autocomplete_ngram():
    return render_template('ngram.html')

# Route to search for books using N-gram predictions
@app.route('/search', methods=['GET'])
def search_ngram():
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])

    try:
        # Generate n-gram predictions
        results = search_books_by_ngram(query)

        return jsonify(results)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

# ------------------- LSTM Autocomplete ------------------- #
# Load the trained LSTM model
model = load_model('model_lstm.h5')

# Load the tokenizer for the LSTM model
with open('tokenizer.json') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

# Function to predict two words using LSTM model
def generate_two_word_prediction(seed_text):
    max_sequence_len = model.input_shape[1] + 1  # Get max sequence length from the model input shape
    
    # Convert the seed text to a token list
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict the first word
    predicted = model.predict(token_list, verbose=0)
    first_word_index = np.argmax(predicted, axis=-1)[0]
    
    first_word = [word for word, index in tokenizer.word_index.items() if index == first_word_index]
    if not first_word:
        return seed_text  # If no prediction, return the seed text
    
    # Append the first predicted word to seed text
    seed_text_with_first_word = seed_text + " " + first_word[0]
    
    # Predict the second word using the updated seed text
    token_list = tokenizer.texts_to_sequences([seed_text_with_first_word])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    second_word_index = np.argmax(predicted, axis=-1)[0]
    
    second_word = [word for word, index in tokenizer.word_index.items() if index == second_word_index]
    if not second_word:
        return seed_text_with_first_word  # If no second prediction, return the current text
    
    # Combine the seed text with the two predicted words
    predicted_phrase = seed_text_with_first_word + " " + second_word[0]
    
    return predicted_phrase

# Search for books using LSTM predictions
def search_books_by_predicted_phrase(predicted_phrase, max_results=10):
    results = []
    for full_title in book_metadata.keys():
        if predicted_phrase in full_title:
            results.append(book_metadata[full_title])
    return results[:max_results]

# Route to render the LSTM autocomplete page
@app.route('/lstm')
def autocomplete_lstm():
    return render_template('lstm.html')

# Route to search for books using LSTM predictions
@app.route('/search_lstm', methods=['GET'])
def search_lstm():
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])

    try:
        # Generate a two-word prediction based on user input
        predicted_phrase = generate_two_word_prediction(query)

        # Search for matching titles in the book metadata
        results = search_books_by_predicted_phrase(predicted_phrase)

        return jsonify(results)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

# ------------------- Choose Autocomplete (LSTM or N-gram) ------------------- #
@app.route('/choose_autocomplete')
def choose_autocomplete():
    return render_template('choose_autocomplete.html')

# Redirect to the choice page
@app.route('/')
def home():
    return redirect('/choose_autocomplete')

if __name__ == '__main__':
    app.run(debug=True)
