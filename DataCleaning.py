
stop_words = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'i', 'you', 'they', 'we', 'this', 'but', 'or', 'not'
}

# Keep domain-relevant words
stop_words -= {''} 

def remove_stop_words(text, stop_words):
    """
    Remove stop words from a text string.
    
    Args:
        text (str): Input text.
        stop_words (set): Set of stop words to remove.
    
    Returns:
        str: Text with stop words removed.
    """
    # Split text into words
    words = text.split()
    # Keep words that aren't stop words
    filtered_words = [word for word in words if word not in stop_words]
    # Join words back into a string
    return ' '.join(filtered_words)


import pandas as pd


df['summary'] = df['summary'].str.lower()


df['summary'] = df['summary'].apply(lambda x: remove_stop_words(x, stop_words))



import spacy

nlp = spacy.blank('en')  # Blank English pipeline
nlp.add_pipe('lemmatizer', config={'mode': 'rule'})  # Rule-based lemmatizer

def lemmatize_text(text):
    """
    Lemmatize a text string using spacy's rule-based lemmatizer.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Lemmatized text.
    """
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc)

df['descriptions'] = df['descriptions'].apply(lemmatize_text)
