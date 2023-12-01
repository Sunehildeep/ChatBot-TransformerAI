import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re
import pickle

class DataProcessor:
    def __init__(self, file_path, max_text_len=80, batch_size=256, buffer_size=20000):
        self.file_path = file_path
        self.max_text_len = max_text_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.tokenizer_input = Tokenizer(filters='', char_level=False, lower=True)
        self.tokenizer_output = Tokenizer(filters='', char_level=False, lower=True)
        self.data = None
        self.progress_train_length = None
        self.progress_test_length = None
        self.train_batches = None
        self.test_batches = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

        original_data = pd.DataFrame({
            'question': self.data['0'],
            'answer': self.data['1']
        })

        additional_data = pd.DataFrame({
            'question': self.data['1'],  # Treat '1' column as questions
            'answer': self.data['2']      # Treat '2' column as answers
        })

        # Concatenate original and additional data
        self.data = pd.concat([original_data, additional_data], ignore_index=True)
        mask = (self.data['question'].str.len() <= self.max_text_len) & (self.data['answer'].str.len() <= self.max_text_len)
        self.data = self.data[mask]

    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F" +
                                   u"\U0001F300-\U0001F5FF" +
                                   u"\U0001F680-\U0001F6FF" +
                                   u"\U0001F1E0-\U0001F1FF" +
                                   u"\U00002500-\U00002BEF" +
                                   u"\U00002702-\U000027B0" +
                                   u"\U000024C2-\U0001F251" +
                                   u"\U0001f926-\U0001f937" +
                                   u"\U00010000-\U0010ffff" +
                                   u"\u200d" +
                                   u"\u2640-\u2642" +
                                   u"\u2600-\u2B55" +
                                   u"\u23cf" +
                                   u"\u23e9" +
                                   u"\u231a" +
                                   u"\u3030" +
                                   u"\ufe0f"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def expand_contractions(self, sentence):
        # Define contraction mapping
        contractions_dict = {
            "ain't": "are not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "I'd": "I would",
            "I'd've": "I would have",
            "I'll": "I will",
            "I'll've": "I will have",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have"
        }

        # Create a regular expression pattern to match contractions
        contraction_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                        flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = contractions_dict.get(match) or contractions_dict.get(match.lower())
            if expanded_contraction is None:
                return match
            return expanded_contraction

        expanded_sentence = contraction_pattern.sub(expand_match, sentence)
        return expanded_sentence

    def preprocess_data(self):
        self.data['question'] = self.data['question'].apply(lambda x: self.remove_emojis(x))
        self.data['answer'] = self.data['answer'].apply(lambda x: self.remove_emojis(x))
        self.data['question'] = self.data['question'].apply(lambda x: self.expand_contractions(x))
        self.data['answer'] = self.data['answer'].apply(lambda x: self.expand_contractions(x))
        self.data['question'] = self.data['question'].apply(lambda x: re.sub(r'\d+', '', x))
        self.data['answer'] = self.data['answer'].apply(lambda x: re.sub(r'\d+', '', x))
        self.data['question'] = self.data['question'].apply(lambda x: re.sub(r'http\S+', '', x))
        self.data['answer'] = self.data['answer'].apply(lambda x: re.sub(r'http\S+', '', x))
        self.data['question'] = self.data['question'].apply(lambda x: re.sub(r'\s+', ' ', x))
        self.data['answer'] = self.data['answer'].apply(lambda x: re.sub(r'\s+', ' ', x))
        self.data['question'] = self.data['question'].apply(lambda x: '<start> ' + x + ' <end>')
        self.data['answer'] = self.data['answer'].apply(lambda x: '<start> ' + x + ' <end>')

    def tokenize_data(self):
        self.tokenizer_input.fit_on_texts(self.data['question'])
        self.tokenizer_output.fit_on_texts(self.data['answer'])

    def split_data(self, test_size=0.2, random_state=42):
        train, test = train_test_split(self.data, test_size=test_size, random_state=random_state)
        return train, test

    def dump_tokenizers(self):
        with open('tokenizer_input.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer_input, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('tokenizer_output.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def encode(self, ques, ans):
        ques = self.tokenizer_input.texts_to_sequences([ques.numpy().decode('utf-8')])[0]
        ans = self.tokenizer_output.texts_to_sequences([ans.numpy().decode('utf-8')])[0]
        return ques, ans

    def set_shapes(self, ques, ans):
        result_ques, result_ans = tf.py_function(self.encode, [ques, ans], [tf.int64, tf.int64])
        tar_inp = result_ans[:-1]
        tar_real = result_ans[1:]
        result_ques.set_shape([None])
        result_ans.set_shape([None])
        tar_inp.set_shape([None])
        tar_real.set_shape([None])
        return (result_ques, tar_inp), tar_real

    def make_batches(self, ds):
        return (
            ds
            .shuffle(self.buffer_size)
            .map(lambda x: self.set_shapes(x['question'], x['answer']))
            .padded_batch(self.batch_size, padded_shapes=(([None], [None]), [None]))
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def process_dataset(self):
        self.load_data()
        self.preprocess_data()
        self.tokenize_data()
        train, test = self.split_data()
        self.dump_tokenizers()

        train_dataset = tf.data.Dataset.from_tensor_slices(dict(train))
        test_dataset = tf.data.Dataset.from_tensor_slices(dict(test))

        self.train_batches = self.make_batches(train_dataset)
        self.test_batches = self.make_batches(test_dataset)

        self.progress_train_length = len(train) * 0.8
        self.progress_test_length = len(train) * 0.2

    
    def get_config(self):
        return {
            'num_layers': 3,
            'd_model': 128,
            'dff': 512,
            'num_heads': 16,
            'dropout_rate': 0.1,
            'input_vocab_size': len(self.tokenizer_input.index_word) + 2,
            'target_vocab_size': len(self.tokenizer_output.index_word) + 2,
        }