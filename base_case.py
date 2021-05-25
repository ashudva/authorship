# Imports
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import keras

# bert-base-uncased pretrained model will be used as base for fine-tuning
model_name = ''

#---------------------------------------------------------------------------#
# Data Preprocessing
#---------------------------------------------------------------------------#
# Instantiate the tokenizer
tokenizer = AutoTokenizer()