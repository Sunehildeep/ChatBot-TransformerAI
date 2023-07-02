import pickle
from model import Transformer, create_masks
import tensorflow as tf
import numpy as np

max_text_len = 80

with open('tokenizer_input.pickle', 'rb') as f:
    tokenizer_input = pickle.load(f)

with open('tokenizer_output.pickle', 'rb') as f:
    tokenizer_output = pickle.load(f)

num_layers = 2
d_model = 256
dff = 512
num_heads = 4
input_vocab_size = len(tokenizer_input.index_word) + 2
target_vocab_size = len(tokenizer_output.index_word) + 2
dropout_rate = 0.2

transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=2048, pe_target=2048, rate=dropout_rate)

optimizer = tf.keras.optimizers.RMSprop()

checkpoint_path = "./checkpoints_test/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


def search(text, model, tokenizer_q, tokenizer_a, width=5, temperature=0.9):
    start_token = [tokenizer_q.word_index['<start>']]
    end_token = [tokenizer_q.word_index['<end>']]

    # All questions have the start and end token
    inp_sentence = start_token + tokenizer_q.texts_to_sequences([text])[0] + end_token
    inp_sentence = tf.keras.preprocessing.sequence.pad_sequences([inp_sentence], maxlen=max_text_len, padding='post')

    # 'answers' start token : 27358
    decoder_input = [tokenizer_a.word_index['<start>']]
    decoder_input = tf.expand_dims(decoder_input, 0)

    for i in range(max_text_len):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_sentence, decoder_input)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(inp_sentence,
                                               decoder_input,
                                               False,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        # apply a softmax to normalize the predictions into a probability distribution
        predictions = tf.nn.softmax(predictions, axis=-1)

        # apply temperature to control the randomness of sampling
        predictions /= temperature

        # use top-k sampling to introduce randomness in choosing the predicted_id
        top_k_predictions = tf.math.top_k(predictions, k=width)
        indices = top_k_predictions.indices.numpy()
        values = top_k_predictions.values.numpy()

        probabilities = values / np.sum(values)

        # choose one of the top k indices based on their probability
        predicted_id = np.random.choice(indices[0][0], p=probabilities[0][0])

        predicted_id = tf.expand_dims([predicted_id], 0)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_a.word_index['<end>']:
            break

        # concatenate the predicted_id to the output which is given to the decoder
        # as its input.
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)


    # Remove the start token from the predictions
    decoder_input = decoder_input[:, 1:]
    return tf.squeeze(decoder_input, axis=0), attention_weights


# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

# #Interference
while True:
    text = input("Enter question: ")
    output, _ = search(text, transformer, tokenizer_input, tokenizer_output)
    print("Answer: {}\n".format(tokenizer_output.sequences_to_texts([output.numpy()])[0]))
