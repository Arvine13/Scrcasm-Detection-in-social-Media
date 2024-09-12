import re
import string

import numpy as np
from nltk import download
from nltk.corpus import stopwords
import nltk
import tqdm
import time

download('stopwords')
download('punkt')

NEGATIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
    "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
    "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is",
    "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
    "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"
}
NEGATIONS_PATTERN = re.compile(r'\b(' + '|'.join(NEGATIONS.keys()) + r')\b')
PUNCTUATIONS = string.punctuation
SINGLE_LETTER_WORDS_PATTERN = re.compile(r'(?<![\w\-])\w(?![\w\-])')
BLANK_SPACES_PATTERN = re.compile(r'\s{2,}|\t')
STOPWORDS = set(stopwords.words('english'))


def handle_negations(text):
    return re.sub(pattern=NEGATIONS_PATTERN, repl='', string=text)


def remove_single_letter_words(text):
    return re.sub(pattern=SINGLE_LETTER_WORDS_PATTERN, repl='', string=text)


def remove_blank_spaces(text):
    return re.sub(pattern=BLANK_SPACES_PATTERN, repl=' ', string=text)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCTUATIONS))


def remove_numbers(text):
    text_list = text.split(' ')
    for tex in text_list:
        if tex.isnumeric():
            text_list.remove(tex)
    return ' '.join(text_list)


def remove_stopwords(text):
    tex = nltk.word_tokenize(text)
    new_sentence = []
    for w in tex:
        if w not in STOPWORDS:
            new_sentence.append(w)
    return ' '.join(new_sentence)


def to_lowercase(text):
    return text.lower()


PREPROCESSING_TECHNIQUES = {
    'Handle Negations': handle_negations,
    'Removing Single Letter Words': remove_single_letter_words,
    'Removing Multiple Spaces': remove_blank_spaces,
    'Removing Punctuation': remove_punctuation,
    'Removing Numbers': remove_numbers,
    'Removing Stopwords': remove_stopwords,
    'Convert To Lower Case': to_lowercase,
}


def preprocess(text):
    for technique in PREPROCESSING_TECHNIQUES.values():
        text = technique(text)
    return text if text != '' else np.NAN


def preprocess_df(df, text_col, log):
    tqdm.tqdm.pandas(desc='[INFO] PreProcessing :')
    log('PreProcessing Steps.')
    log(('\n'.join(['\t [{0}] {1}'.format(i+1, tec) for i, tec in enumerate(PREPROCESSING_TECHNIQUES.keys())]), True))
    time.sleep(0.1)
    df[text_col] = df[text_col].progress_apply(lambda x: preprocess(x))
    log('Data PreProcessed!')
    return df
