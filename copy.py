from pathlib import Path

data_dir = Path('data/C50')
test_dir = data_dir / 'test'
train_dir = data_dir / 'train'

test_sub_dirs = test_dir.iterdir()
train_sub_dirs = list(train_dir.iterdir())



for i, author in enumerate(test_sub_dirs):
    for file in list(author.iterdir())[:-10]:
        # print(len(list(author.iterdir())[:-10]))
        file_name = file.name
        dest = train_sub_dirs[i] / file_name 
        # print(f"dest: {dest}, file: {file_name}, src: {file}")
        file.replace(dest)