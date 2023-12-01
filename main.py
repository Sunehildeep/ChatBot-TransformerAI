import tensorflow as tf
from preprocess import DataProcessor
from model import Transformer, create_masks
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.utils import Progbar
import numpy as np

train_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

test_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=5000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class CustomTrainer:
    def __init__(self, data_processor, transformer, optimizer, loss_object):
        self.data_processor = data_processor
        self.transformer = transformer
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        self.ckpt_manager = tf.train.CheckpointManager(tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer),
                                                       "./checkpoints_test/train", max_to_keep=5)

    @tf.function(input_signature=train_signature)
    def train_step(self, inp, tar_inp, tar_real):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.transformer.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    @tf.function(input_signature=test_signature)
    def test_step(self, inp, tar_inp, tar_real):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp)
        predictions, _ = self.transformer(
            inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = self.loss_function(tar_real, predictions)
        self.test_loss(loss)
        self.test_accuracy(tar_real, predictions)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def search(self, text, model, tokenizer_q, tokenizer_a, width=5, temperature=0.9):
        start_token = [tokenizer_q.word_index['<start>']]
        end_token = [tokenizer_q.word_index['<end>']]

        # All questions have the start and end token
        inp_sentence = start_token + \
            tokenizer_q.texts_to_sequences([text])[0] + end_token
        inp_sentence = tf.keras.preprocessing.sequence.pad_sequences(
            [inp_sentence], maxlen=self.data_processor.max_text_len, padding='post')

        # 'answers' start token : 27358
        decoder_input = [tokenizer_a.word_index['<start>']]
        decoder_input = tf.expand_dims(decoder_input, 0)

        for i in range(self.data_processor.max_text_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp_sentence, decoder_input)

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
            predicted_id = np.random.choice(
                indices[0][0], p=probabilities[0][0])

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

    def train(self, EPOCHS=100):
        for epoch in range(EPOCHS):
            print("\nEpoch {}/{}".format(epoch + 1, EPOCHS))
            pb = Progbar(self.data_processor.progress_train_length + self.data_processor.progress_test_length,
                         stateful_metrics=['loss', 'acc', 'val_loss', 'val_acc'])

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, ((inp, tar_inp), tar_real)) in enumerate(self.data_processor.train_batches):
                self.train_step(inp, tar_inp, tar_real)
                train_values = [('loss', self.train_loss.result()),
                                ('acc', self.train_accuracy.result())]
                pb.add(inp.shape[0], values=train_values)

            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for (_, ((inp, tar_inp), tar_real)) in enumerate(self.data_processor.test_batches):
                self.test_step(inp, tar_inp, tar_real)
                val_values = [('val_loss', self.test_loss.result()),
                              ('val_acc', self.test_accuracy.result())]
                pb.add(inp.shape[0], values=val_values)

            if epoch % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(
                    epoch + 1, ckpt_save_path))

            output, _ = self.search("how are you?", self.transformer,
                                    self.data_processor.tokenizer_input, self.data_processor.tokenizer_output)
            print("Question: How are you?")
            print("Answer: {}".format(
                self.data_processor.tokenizer_output.sequences_to_texts([output.numpy()])[0]))

if __name__ == "__main__":
    data_processor = DataProcessor(file_path='reddit.csv')
    data_processor.process_dataset()

    config = data_processor.get_config()

    custom_schedule = CustomSchedule(config['d_model'])
    optimizer = tf.keras.optimizers.RMSprop(custom_schedule)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    custom_trainer = CustomTrainer(data_processor, Transformer(config['num_layers'], config['d_model'], config['num_heads'], config['dff'],
                                   config['input_vocab_size'], config['target_vocab_size'], pe_input=2048, pe_target=2048, rate=config['dropout_rate']), optimizer, loss_object)

    custom_trainer.train()
