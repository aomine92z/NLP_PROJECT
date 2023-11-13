import pandas as pd
import re
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import plotly.express as px

lemmatizer = WordNetLemmatizer() # Initialize the WordNet lemmatizer

class Model:
    def __init__(self, X, y, model_architecture, random_seed=42, test_size=0.15) -> None:
        self.X = X
        self.y = y
        self.model_instance = model_architecture
        self.random_seed = random_seed
        self.test_size = test_size  

        # Define the pipeline
        self.pipeline = Pipeline([
            ('classifier', model_architecture)
        ])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y) # train test split using the above X, y, test_size and random_state
    
    def fit(self):
        # fit self.pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self):
        return self.pipeline.predict(self.X_test)
    
    def predict_proba(self):
        return self.pipeline.predict_proba(self.X_test)

    def report(self, y_true, y_pred, class_labels):
        # the report function as defined previously
        print(classification_report(y_true, y_pred, target_names=class_labels))
        
        # Create the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Display the confusion matrix using Plotly
        confusion_matrix_kwargs = dict(
            text_auto=True, 
            title="Confusion Matrix", width=1000, height=800,
            labels=dict(x="Predicted", y="True Label"),
            color_continuous_scale='Blues'
        )
        fig = px.imshow(
            cm,
            **confusion_matrix_kwargs,
            x=class_labels,
            y=class_labels
        )
        fig.show()

def clean_text(text: str) -> str:
    """
        Perform basic text cleaning operations.
        - Remove special characters (except spaces and digits)
        - Convert text to lowercase

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.

        Example:
            >>> clean_text("This is an example with special characters: !@#$")
            "this is an example with special characters"
    """
    text = re.sub(r'[^\w\s]', '', text.replace('\n', ' ')) # Remove special characters (except spaces and digits)
    text = text.lower() # Convert to lowercase
    return text # Return cleaned text

def correct_spellings(text: str) -> str:
    """
        Correct spelling errors in the input text using a spell checker.

        This function identifies and corrects spelling errors in the input text by utilizing a spell checker
        (presumably the `spell` object). It splits the input text into words, identifies misspelled words,
        and replaces them with their corrected versions. The corrected text is then returned.

        Args:
            text (str): The input text with possible spelling errors.

        Returns:
            str: A new string with spelling errors corrected.

        Example:
            >>> correct_spellings("Ths is an exmple of misspeled wrds.")
            "This is an example of misspelled words."
    """
    spell = SpellChecker()
    corrected_text = []

    # Use the spell.unknown() function to identify misspelled words in the text
    misspelled_words = spell.unknown(text.split())

    # Iterate through the text
    for word in text.split():
        # If the current word is misspelled
        if word in misspelled_words:
            # Get the corrected version
            correction = spell.correction(word)
            if correction is not None:
                # Append the corrected version to corrected_text
                corrected_text.append(correction)
        else:
            # Otherwise, append the original word
            corrected_text.append(word)

    # Return the joined version of the text
    return ' '.join(corrected_text)

def get_top_words(data, nwords):
    """
        Get the top 12 most frequent words in the lyrics.

        Args:
            data (pd.DataFrame): The DataFrame containing the lyrics.

        Returns:
            list: The top nwords most frequent words.

        Example:
            >>> lyrics_data = pd.DataFrame({"Lyric": ["this is a test", "another test"]})
            >>> get_top_words(lyrics_data)
            [('test', 2), ('this', 1), ('is', 1), ('a', 1), ('another', 1)]
    """
    all_lyrics = ' '.join(data['Lyric']) # Extract all lyrics into a single string
    words = all_lyrics.split() # Tokenize the combined lyrics into words
    word_counter = Counter(words) # Use Counter to count the frequency of each word
    FREQWORDS = word_counter.most_common(nwords) # Get the top 12 most frequent words

    return FREQWORDS # Return TOP 12 most frequent words

def remove_freqwords(FREQWORDS, text: str) -> str:
    """
        Removes freqwords and frequent words from the input text

        Args: 
            text (str): The input text from which freqwords will be removed

        Returns:
            str: A new string without freqwords
    """
    
    # Convert the string to a list of words using split and remove the stopwords

    filtered_words = [] # List comprehension
    text_split = text.split(' ') # Split the text according the spaces
    for word in text_split: # For each word
        if word not in FREQWORDS: # If the word IS NOT in STOPWORDS
            filtered_words.append(word) # Keep this word in the final text

    return " ".join(filtered_words)  # Return the text without stopwords

def remove_stopwords(text: str) -> str:
    """
        Removes stopwords and frequent words from the input text

        Args: 
            text (str): The input text from which stopwords will be removed

        Returns:
            str: A new string without stopwords
    """
    STOPWORDS = set(stopwords.words('english')).union(set(stopwords.words('french'))) # Get stopwords in english and french

    # Convert the string to a list of words using split and remove the stopwords

    filtered_words = [] # List comprehension
    text_split = text.split(' ') # Split the text according the spaces
    for word in text_split: # For each word
        if word not in STOPWORDS: # If the word IS NOT in STOPWORDS
            filtered_words.append(word) # Keep this word in the final text

    return " ".join(filtered_words)  # Return the text without stopwords
  
def lemmatize_words(text: str) -> str:
    """
        Apply lemmatization to the input string, considering words' POS tags.

        This function lemmatizes words in the input string based on their POS (Part-of-Speech) tags.
        
        Args:
            text (str): The input text to be lemmatized.

        Returns:
            str: A new string with lemmatized words.
    """
    # Initialize a mapping of POS tags to WordNet tags
    wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }
    pos_tagged_text = nltk.pos_tag(nltk.word_tokenize(text)) # Get the POS tags of every word in the input
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0].upper(), wordnet.NOUN)) for word, pos in pos_tagged_text] # Lemmatize the words based on their POS tags
    
    return ' '.join(lemmatized_words) # Return the lemmatized text

def load_preprocessed_data(filepath, artists_to_keep):
    """
        Apply the full preprocessing pipeline to the input DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing the lyrics.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.

        Example:
            >>> load_preprocessed_data(pd.DataFrame({"Lyric": ["This is a test.", "Another test!"]}))
            pd.DataFrame
    """
    data = pd.read_csv(filepath, usecols=["ALink", "SName", "Lyric"]) # Import the data in a pandas dataframe
    data = data.rename(columns={'ALink': 'Artist', 'SName': 'Title'}) # Rename the columns
    
    data = data.drop(data.loc[data['Lyric'].isna()].index) # Remove rows with missing values
    data['Artist'] = data['Artist'].str.replace('/', '')  # Remove '/' in the name of the artist
    data['Artist'] = data['Artist'].apply(lambda x: x.lower().replace('-', ' ')) # Lowerize first letter of each name, replace '-' with ' '
    data = data.drop_duplicates(subset=['Lyric']) # Remove duplicates in the Lyric column
    
    data = data[data["Artist"].isin(artists_to_keep)] # Keep only the artists of interest for the study
    
    data = data.reset_index(drop=True) # Reset the index and drop the previous one
    data["Lyric_Length"] = data["Lyric"].str.len() # Add a column length to know the size of a song in characters --> New feature
    
    # Apply the cleaning operations
    data['Lyric'] = data['Lyric'].apply(clean_text) # Clean Lyrics
    data['Title'] = data['Title'].apply(clean_text) # Clean Titles

    # Apply lemmatization
    data['Lyric'] = data['Lyric'].apply(lemmatize_words) # Lemmatize Lyrics
    data['Title'] = data['Title'].apply(lemmatize_words) # Lemmatize Titles

    # Apply filtering
    data['Lyric'] = data.apply(lambda row: remove_stopwords(row['Lyric']), axis=1) # Remove  StopWords from Lyrics
    data['Title'] = data.apply(lambda row: remove_stopwords(row['Title']), axis=1) # Remove  StopWords from Titles

    FREQWORDS = get_top_words(data, 12) # Get the most frequent words
    data['Lyric'] = data['Lyric'].apply(lambda lyric: remove_freqwords(FREQWORDS, lyric)) # Remove Most Frequent Words from Lyrics

    return data # Return preprocessed loaded data