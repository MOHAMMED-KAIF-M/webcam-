import os
from huggingface_hub import snapshot_download

os.makedirs('models1', exist_ok=True)
print('Downloading action recognition model into models1...')
path = snapshot_download(
    repo_id='MCG-NJU/videomae-base-finetuned-kinetics',
    local_dir='models1',
    local_dir_use_symlinks=False,
)
print('Downloaded to', path)
