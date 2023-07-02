from preprocess import *
from model import *
from tensorflow.keras.utils import Progbar

num_layers = 2
d_model = 256
dff = 512
num_heads = 4
input_vocab_size = len(tokenizer_input.index_word) + 2
target_vocab_size = len(tokenizer_output.index_word) + 2
dropout_rate = 0.2

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
optimizer = tf.keras.optimizers.RMSprop(CustomSchedule(d_model))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Transformer
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=2048, pe_target=2048, rate=dropout_rate)

checkpoint_path = "./checkpoints_test/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

#The inp, tar_inp, tar_real are creatd using to_tensor() function defined in preprocess.py
# Create a signature
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar_inp, tar_real):

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp) 
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

test_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=test_step_signature)
def test_step(inp, tar_inp, tar_real):

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp) 
    predictions, _ = transformer(inp, tar_inp, 
                                 False, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

    test_loss(loss)
    test_accuracy(tar_real, predictions)

EPOCHS = 200
metrics_names = ['loss', 'acc', 'val_loss', 'val_acc']
train_loss.reset_states()
train_accuracy.reset_states()
test_loss.reset_states()
test_accuracy.reset_states()

class EarlyStoppping(tf.keras.callbacks.Callback):
    def __init__(self, model, patience=0, min_delta=0):
        super(EarlyStoppping, self).__init__()
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if np.less(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            print("Early stopping ({}/{})".format(self.wait, self.patience))
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

early_stopping = EarlyStoppping(transformer, patience=10, min_delta=0.001)

if __name__ == "__main__":
    # Training
    for epoch in range(EPOCHS):
        print("\nEpoch {}/{}".format(epoch+1, EPOCHS))
        pb = Progbar(progress_train_length + progress_test_length, stateful_metrics=metrics_names)

        train_loss.reset_states()  # Reset train loss at the start of each epoch
        train_accuracy.reset_states()  # Reset train accuracy at the start of each epoch

        # inp -> question, tar -> answer
        for (batch, ((inp, tar_inp), tar_real)) in enumerate(train_batches):
        
            train_step(inp, tar_inp, tar_real)

            train_values = [('loss', train_loss.result()), ('acc', train_accuracy.result())]
            pb.add(inp.shape[0], values=train_values)  # Use inp.shape[0] to get the actual number of examples processed

        test_loss.reset_states()  # Reset validation loss after the training loop
        test_accuracy.reset_states()  # Reset validation accuracy after the training loop

        for (batch, ((inp, tar_inp), tar_real))in enumerate(test_batches):
            test_step(inp, tar_inp, tar_real)

            val_values = [('val_loss', test_loss.result()), ('val_acc', test_accuracy.result())]
            pb.add(inp.shape[0], values=val_values)  # Use inp.shape[0] to get the actual number of examples processed

        early_stopping.on_epoch_end(epoch, logs={'val_loss': test_loss.result()})

        #Early stopping using early_stopping callback
        if early_stopping.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (epoch))
            break

        # output, _ = search("how are you?", transformer, tokenizer_input, tokenizer_output)
        # print("Question: How are you?")
        #Use keras index_word
        # print("Answer: {}".format("".join([tokenizer_output.index_word[i] for i in output.numpy() if i < len(tokenizer_output.index_word)])))
        # print("Answer: {}".format(tokenizer_output.sequences_to_texts([output.numpy()])[0]))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))