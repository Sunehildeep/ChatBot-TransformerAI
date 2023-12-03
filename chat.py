import pickle
from model import Transformer, create_masks
import tensorflow as tf
import numpy as np

class ChatBot:
    def __init__(self, tokenizer_input_path='tokenizer_input.pickle', tokenizer_output_path='tokenizer_output.pickle',
                 num_layers=3, d_model=128, dff=512, num_heads=16, dropout_rate=0.1, max_text_len=80):
        self.max_text_len = max_text_len
        self.load_tokenizers(tokenizer_input_path, tokenizer_output_path)
        self.build_model(num_layers, d_model, dff, num_heads, dropout_rate)
        self.load_checkpoint()

    def load_tokenizers(self, tokenizer_input_path, tokenizer_output_path):
        with open(tokenizer_input_path, 'rb') as f:
            self.tokenizer_input = pickle.load(f)

        with open(tokenizer_output_path, 'rb') as f:
            self.tokenizer_output = pickle.load(f)

    def build_model(self, num_layers, d_model, dff, num_heads, dropout_rate):
        input_vocab_size = len(self.tokenizer_input.index_word) + 2
        target_vocab_size = len(self.tokenizer_output.index_word) + 2
        self.transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                                       pe_input=2048, pe_target=2048, rate=dropout_rate)

    def load_checkpoint(self, optimizer=tf.keras.optimizers.RMSprop()):
        checkpoint_path = "./checkpoints_test/train"
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def search(self, text, width=5, temperature=0.9):
        start_token = [self.tokenizer_input.word_index['<start>']]
        end_token = [self.tokenizer_input.word_index['<end>']]

        inp_sentence = start_token + self.tokenizer_input.texts_to_sequences([text])[0] + end_token
        inp_sentence = tf.keras.preprocessing.sequence.pad_sequences([inp_sentence], maxlen=self.max_text_len,
                                                                     padding='post')

        decoder_input = [self.tokenizer_output.word_index['<start>']]
        decoder_input = tf.expand_dims(decoder_input, 0)

        for i in range(self.max_text_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_sentence, decoder_input)

            predictions, attention_weights = self.transformer(inp_sentence, decoder_input, False, enc_padding_mask,
                                                               combined_mask, dec_padding_mask)

            predictions = predictions[:, -1:, :]
            predictions = tf.nn.softmax(predictions, axis=-1)
            predictions /= temperature

            top_k_predictions = tf.math.top_k(predictions, k=width)
            indices = top_k_predictions.indices.numpy()
            values = top_k_predictions.values.numpy()

            probabilities = values / np.sum(values)

            predicted_id = np.random.choice(indices[0][0], p=probabilities[0][0])

            predicted_id = tf.expand_dims([predicted_id], 0)

            if predicted_id == self.tokenizer_output.word_index['<end>']:
                break

            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

        decoder_input = decoder_input[:, 1:]
        return tf.squeeze(decoder_input, axis=0), attention_weights

    def chat(self):
        while True:
            text = input("Enter your prompt: ")
            output, _ = self.search(text)
            answer = self.tokenizer_output.sequences_to_texts([output.numpy()])[0]
            print(f"Bot: {answer}\n")


if __name__ == '__main__':
    chatbot = ChatBot()
    chatbot.chat()
