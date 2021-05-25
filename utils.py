import matplotlib.pyplot as plt
from keras import models, layers
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

AUTOTUNE = tf.data.AUTOTUNE

def get_text(ds:tf.data.Dataset) -> tuple:
    """
    Get the text from a tf.data.Dataset object
    Args:
        ds: input dataset
    Return:
        (text, labels)
    """
    texts = []
    labels = []
    for batch, label in ds:
        labels.extend(label.numpy())
        for text in batch:
            texts.append(str(text.numpy()))
    return (texts, labels)


def prepare_unbatched(ds, tokenizer):
    """
    Perform following ops in the input dataset
    1. Get the text
    2. Encode text
    3. Make tf.data.Dataset
    4. Shuffle
    5. Prefetch and Cache
    """
    text, labels = get_text(ds)
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True
    )

    ds = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels
    ))
    ds = ds.shuffle(1000)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

def prepare_batched(ds, tokenizer, batch_size=16):
    """
    Perform following ops in the input dataset
    1. Get the text
    2. Encode text
    3. Make tf.data.Dataset
    4. Shuffle
    5. Batch
    5. Prefetch and Cache
    """
    text, labels = get_text(ds)
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True
    )

    ds = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels
    ))
    ds = ds.shuffle(1000).batch(batch_size)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)



def save_object(obj: object, file_path: Path) -> None:
    """
    Save a python object to the disk and creates the file if does not exists already.
    Args:
        file_path - Path object for pkl file location
        obj       - object to be saved
    
    Returns:
        None
    """
    if not file_path.exists():
        file_path.touch()
        print(f"pickle file {file_path.name} created successfully!")
    else:
        print(f"pickle file {file_path.name} already exists!")

    with file_path.open(mode='wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"object {type(obj)} saved to file {file_path.name}!")


def load_object(file_path: Path) -> object:
    """
    Loads the pickle object file from the disk.
    Args:
        file_path - Path object for pkl file location
    
    Returns:
        object
    """
    if file_path.exists():
        with file_path.open(mode='rb') as file:
            print(f"loaded object from file {file_path.name}")
            return pickle.load(file)
    else:
        raise FileNotFoundError


def vectorize_sequence(sequences, dimension=10000):
    """
    Convert sequences into one-hot encoded matrix of dimension [len(sequence), dimension]
    
    Args: sequences - ndarray of shape [samples, words]
          dimension = number of total words in vocab
    Return: vectorized sequence of shape [samples, one-hot-vecotor]
    """
    # Create all-zero matrix
    results = np.zeros((len(sequences), dimension))
    
    for (i, sequence) in enumerate(sequences):
        results[i, sequence] = 1.
        
    return results


def plot_history(history):
    """
    Plots the history of training of a model during epochs
    
    Args: history - model history
    
    Plots:
    1. Training and Validation Loss
    2. Training and Validation Accuracy
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.epoch, history.history.get(
        'loss'), "o", label='train loss')
    ax1.plot(history.epoch, history.history.get(
        'val_loss'), '-', label='val loss')
    ax2.plot(history.epoch, history.history.get(
        'acc'), 'o', label='train acc')
    ax2.plot(history.epoch, history.history.get(
        'val_acc'), '-', label='val acc')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax1.set_title("Loss")
    ax2.set_title("Accuracy")
    f.suptitle("Training and Validation History")
    ax1.legend()
    ax2.legend()

def plot_mae_history(history):
    """
    Plots the history of training of a model during epochs
    
    Args: history - model history
    
    Plots:
    1. Training and Validation Loss
    2. Training and Validation Accuracy
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.epoch, history.history.get(
        'loss'), "o", label='train loss')
    ax1.plot(history.epoch, history.history.get(
        'val_loss'), '-', label='val loss')
    ax2.plot(history.epoch, history.history.get(
        'mae'), 'o', label='train mae')
    ax2.plot(history.epoch, history.history.get(
        'val_mae'), '-', label='val mae')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mae")
    ax1.set_title("Loss")
    ax2.set_title("MAE")
    f.suptitle("Training and Validation History")
    ax1.legend()
    ax2.legend()

def buil_model(nfeatures):
    """
    Returns a model with input_shape as nfeatures
    """

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=nfeatures))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    return model
