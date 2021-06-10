from tqdm.auto import tqdm
from transformers import AutoTokenizer

from tokenizers import Tokenizer
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
from pathlib import Path


################
# Creating data files to encode
################
train_dir = str(input("Training directory path: "))
model = str(input("Model name: "))
data_dir = Path(train_dir)
fnames = list(data_dir.glob('*/*.txt'))
nfiles = len(fnames)
files = [txt.read_text(encoding='utf-8') for txt in tqdm(fnames, desc="Reading files", total=nfiles,)]


################
# Setup tokenizer
################

# Instantiate tokenizer with AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model)


################
# Training and saving
################

# Train the tokenizer
tokenizer.train_from_iterator(files, trainer, length=nfiles)

# Save trained tokenizer
tokenizer.save('tokenizers/gutenberg.json')