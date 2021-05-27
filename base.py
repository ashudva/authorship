######## Imports ########
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils import load_object
from utils import save_object
from utils import plot_history
from keras import models, layers
from keras.optimizers import RMSprop
from keras.preprocessing import text_dataset_from_directory
from keras.layers.experimental.preprocessing import TextVectorization


######## Import Data ########
ds_dir = Path('data/C50/')
train_dir = ds_dir / 'train'
test_dir = ds_dir / 'test'
seed = 123
batch_size = 32

# Prepare batched dataset subsets
train_ds = text_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    seed=seed,
    shuffle=True,
    batch_size=batch_size
)

val_ds = text_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    seed=seed,
    shuffle=True,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation')

test_ds = text_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    seed=seed,
    shuffle=True,
    validation_split=0.2,
    subset='training',
    batch_size=128)

# get class_names for later use
class_names = train_ds.class_names
class_names = np.asarray(class_names)
print(f'sample class names: {class_names[:4]}')

######## Text Vectorization ########

### Define vectorization layers
VOCAB_SIZE = 34000
MAX_LEN = 1400

vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LEN
)

# Train the layers to learn a vocab
train_text = train_ds.map(lambda text, lables: text)
vectorize_layer.adapt(train_text)


# Save the vocabulary to disk
# Run this cell for the first time only
vocab = vectorize_layer.get_vocabulary()
vocab_path = Path('vocab/vocab_C50.pkl')
save_object(vocab, vocab_path)
vocab_len = len(vocab)
print(f"vocab size of vectorizer: {vocab_len}")

######## Configure datasets for performance ########


def vectorize(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds):
    return ds.cache().prefetch(buffer_size=AUTOTUNE)


train_ds = train_ds.map(vectorize)
val_ds = val_ds.map(vectorize)
test_ds = test_ds.map(vectorize)

train_ds = prepare(train_ds)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

######## Parse GloVe pretrained embeddings ########

# Parse the weights
emb_dim = 100
glove_file = Path(f'vocab/glove/glove.6B.{emb_dim}d.txt')
emb_index = {}
with glove_file.open(encoding='utf-8') as f:
    for line in f.readlines():
        values = line.split()
        word = values[0]
        coef = values[1:]
        emb_index[word] = coef

# Getting embedding weights
vocab = load_object(Path('vocab/vocab_C50.pkl'))
emb_matrix = np.zeros((VOCAB_SIZE, emb_dim))
for index, word in enumerate(vocab):
    # get coef of word
    emb_vector = emb_index.get(word)
    if emb_vector is not None:
        emb_matrix[index] = emb_vector
print(f"Shape embedding matrix: {emb_matrix.shape}")


######## Model Definition ########
keras.backend.clear_session()
lstm_model = models.Sequential([
    layers.Embedding(VOCAB_SIZE, emb_dim, input_shape=(MAX_LEN,)),
    layers.Conv1D(256, 11, activation='relu'),
    layers.MaxPooling1D(7),
    layers.Dropout(0.4),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(3),
    layers.Dropout(0.2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(50, activation='softmax')
])

lstm_model.summary()


######### Load pretrained Embeddings ########
lstm_model.layers[0].set_weights([emb_matrix])
lstm_model.layers[0].trainable = False
lstm_model.summary()


######## Training ########

optim = RMSprop(lr=1e-4)

lstm_model.compile(
    loss='CategoricalCrossentropy',
    optimizer=optim,
    metrics=['acc']
)
lstm_history = lstm_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)

# Evaluate the model on test dataset
lstm_model.evaluate(test_ds)

# plot the history of training
plot_history(
    lstm_history,
    save_path=Path('plots/base.jpg'),
    model_name="ConvnetBiLSTM"
)
