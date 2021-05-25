import tokenizers
from utils import load_object, save_object
from tokenizers import Tokenizer
from pathlib import Path

tokenize_file = Path('tokenizers/gutenberg.json').absolute()
tokenizer = None
if tokenize_file.exists():
    tokenizer = Tokenizer(str(tokenize_file))
    print("loaded tokenizer successfully!")
else:
    raise FileNotFoundError

files = 