import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

BATCH_SIZE = 256
BUFFER_SIZE = 20000
max_text_len = 80

# Load and preprocess the dataset
data = pd.read_csv('reddit.csv')

data['question'] = data['0']
data['answer'] = data['1']

data = pd.DataFrame({'question': data['question'], 'answer': data['answer']})

print(data.head())
mask = (data['question'].str.len() <= max_text_len) & (data['answer'].str.len() <= max_text_len)
data = data[mask]

# Remove emojis
import re
def remove_emojis(text):
    # Unicode range for emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # Chinese/Japanese/Korean characters
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def expand_contractions(sentence):
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

data['question'] = data['question'].apply(lambda x: remove_emojis(x))
data['answer'] = data['answer'].apply(lambda x: remove_emojis(x))

# Remove contractions
data['question'] = data['question'].apply(lambda x: expand_contractions(x))
data['answer'] = data['answer'].apply(lambda x: expand_contractions(x))

# Remove numbers
data['question'] = data['question'].apply(lambda x: re.sub(r'\d+', '', x))
data['answer'] = data['answer'].apply(lambda x: re.sub(r'\d+', '', x))

# Remove links
data['question'] = data['question'].apply(lambda x: re.sub(r'http\S+', '', x))
data['answer'] = data['answer'].apply(lambda x: re.sub(r'http\S+', '', x))

# Remove extra spaces
data['question'] = data['question'].apply(lambda x: re.sub(r'\s+', ' ', x))
data['answer'] = data['answer'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Add start and end tokens to sequences
data['question'] = data['question'].apply(lambda x: '<start> ' + x + ' <end>')
data['answer'] = data['answer'].apply(lambda x: '<start> ' + x + ' <end>')

# Tokenize, filter and pad sentences
tokenizer_input = Tokenizer(filters='', char_level=False, lower=True)
tokenizer_input.fit_on_texts(data['question'])

tokenizer_output = Tokenizer(filters='', char_level=False, lower=True)
tokenizer_output.fit_on_texts(data['answer'])

# Just for testing - char level
# #Set index_word 0 to <pad>
# tokenizer_input.index_word[0] = '<pad>'
# tokenizer_output.index_word[0] = '<pad>'

# tokenizer_input.word_index['<start>'] = len(tokenizer_input.word_index)+1
# tokenizer_input.word_index['<end>'] = len(tokenizer_input.word_index)+1
# tokenizer_output.word_index['<start>'] = len(tokenizer_output.word_index)+1
# tokenizer_output.word_index['<end>'] = len(tokenizer_output.word_index)+1

# #Set <start> and <end> to max_index
# tokenizer_input.index_word[tokenizer_input.word_index['<start>']] = '<start>'
# tokenizer_input.index_word[tokenizer_input.word_index['<end>']] = '<end>'
# tokenizer_output.index_word[tokenizer_output.word_index['<start>']] = '<start>'
# tokenizer_output.index_word[tokenizer_output.word_index['<end>']] = '<end>'

# #Set word_index <pad> to 0
# tokenizer_input.word_index['<pad>'] = 0
# tokenizer_output.word_index['<pad>'] = 0


# Split into train and test using skleanr
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)

progress_train_length = len(train)
progress_test_length = len(test)

# Create TensorFlow Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices(dict(train))
test_dataset = tf.data.Dataset.from_tensor_slices(dict(test))

#Length of the batches
print(f"Length of the batches: {len(train_dataset)}")
print(f"Length of the batches: {len(test_dataset)}")

def encode(ques, ans):
    # Just for testing - char level
    #Add start token then encode then add end token 
    # ques = [tokenizer_input.word_index['<start>']] + tokenizer_input.texts_to_sequences([ques.numpy().decode('utf-8')])[0] + [tokenizer_input.word_index['<end>']]
    # ans = [tokenizer_output.word_index['<start>']] + tokenizer_output.texts_to_sequences([ans.numpy().decode('utf-8')])[0] + [tokenizer_output.word_index['<end>']]
    ques = tokenizer_input.texts_to_sequences([ques.numpy().decode('utf-8')])[0]
    ans = tokenizer_output.texts_to_sequences([ans.numpy().decode('utf-8')])[0]
    return ques, ans

def set_shapes(ques, ans):
    result_ques, result_ans = tf.py_function(encode, [ques, ans], [tf.int64, tf.int64])
    tar_inp = result_ans[:-1]
    tar_real = result_ans[1:]
    result_ques.set_shape([None])
    result_ans.set_shape([None])
    tar_inp.set_shape([None])
    tar_real.set_shape([None])
    return (result_ques, tar_inp), tar_real


def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .map(lambda x: set_shapes(x['question'], x['answer']))
        .padded_batch(BATCH_SIZE, padded_shapes=(([None], [None]), [None]))
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

train_batches = make_batches(train_dataset)
test_batches = make_batches(test_dataset)

#Dump the tokenizer
import pickle
with open('tokenizer_input.pickle', 'wb') as handle:
    pickle.dump(tokenizer_input, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tokenizer_output.pickle', 'wb') as handle:
    pickle.dump(tokenizer_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    