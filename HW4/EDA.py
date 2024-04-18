# Import necessary libraries
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def load_text(filename):
    """Load text from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def clean_text(text):
    """Clean and preprocess the text."""
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabet characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def get_word_frequency(text):
    """Calculate word frequency, excluding stopwords."""
    words = text.split()
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    # Count word frequencies
    word_counts = Counter(filtered_words)
    return word_counts

def main():
    # Load and clean the text
    text = load_text('assets/Brave_New_World_Aldous_Huxley_djvu.txt')
    cleaned_text = clean_text(text)
    
    # Get word frequency
    word_freq = get_word_frequency(cleaned_text)
    
    # Print total word count and the 10 most common words
    total_words = len(cleaned_text.split())
    print(f'Total number of words: {total_words}')
    print('Most common words:')
    for word, freq in word_freq.most_common(10):
        print(f'{word}: {freq}')

# Run the main function
if __name__ == '__main__':
    main()
