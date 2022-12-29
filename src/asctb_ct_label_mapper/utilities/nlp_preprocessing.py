import contractions, re, nltk

from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nlp_models():
    print(f'Downloading NLP models required for preprocessing...')
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    print(f'Models downloaded and ready for use!')



def remove_whitespaces(word):
    return word.replace(' ', '')

def expand_word_contractions(word):
    return contractions.fix(word)

def replace_special_chars(word):
    return re.sub('[^a-zA-Z0-9]*', '', word)

def convert_number_to_word(word):
    return num2words(word) if word.isdigit() else word

def make_lowercase(word):
    return word.lower()

def get_root_word(word):
    l = WordNetLemmatizer()
    return l.lemmatize(word)

def is_not_stopword(word):
    return word not in set(stopwords.words('english')) and word!='NaN' 


def execute_nlp_pipeline(word):
    PIPELINE_STEPS = [
        remove_whitespaces,
        expand_word_contractions,
        replace_special_chars,
        convert_number_to_word,
        make_lowercase,
        get_root_word
    ]
    for i, func in enumerate(PIPELINE_STEPS):
        word = func(word)
    return word




def get_asctb_embedding(row, sentence_encoding_model, verbose=False):
    """Performs basic NLP preprocessing tasks with stopwords removed, and returns an embedding of the unique sentence produced.

    Args:
        sentence_encoding_model (SentenceTransformer): Sentence encoder that transforms input cleaned sentence into 768 dimensions.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        np.ndarray: 768x1 EMBEDDING
    """
    ct_id = row['CT_ID']
    ct_label = row['CT_NAME']
    asctb_all_text = row['all_text']
    if verbose:  print(f'CT-ID={ct_id}, CT-LABEL={ct_label}, all_text={asctb_all_text}')
    all_text = []
    unique_words = set()

    for word in asctb_all_text.split(' '):
        cleaned_word = execute_nlp_pipeline(word)
        if verbose:  print(word, cleaned_word)
        if cleaned_word not in unique_words and is_not_stopword(word):
            all_text.append(cleaned_word)
            if verbose:  print(f'All-Text = {all_text}')
            unique_words.add(cleaned_word)
    embedding = sentence_encoding_model.encode(' '.join(all_text))
    return embedding