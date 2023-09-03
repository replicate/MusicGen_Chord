import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import torch.nn as nn
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

df = pd.read_parquet('/mnt/nvme/LP-MusicCaps_vocalless/train0_tagdropped.parquet', engine='pyarrow') 

model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(duration=30)

# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples

target_path = Path("/home/sake/LP-MusicCaps_MSD_CaptionAnalysis")

caption_idx = ["tag", "caption_writing", "caption_summary", "caption_paraphrase", "caption_attribute_prediction"]

rand_idxs = random.sample(range(1, len(df)), 4)

descriptions = []
dfs = []
paths = []

for idx in rand_idxs:
    dfs.append(df.iloc[idx])
    tags = df.iloc[idx].tag
    tag_str = ""
    for tag in tags:
        tag_str = tag_str + tag + ", "
    descriptions.append(tag_str[:-2])
    descriptions.append(df.iloc[idx].caption_writing)
    descriptions.append(df.iloc[idx].caption_summary)
    descriptions.append(df.iloc[idx].caption_paraphrase)
    descriptions.append(df.iloc[idx].caption_attribute_prediction)
    paths.append(df.iloc[idx].path)

print(len(descriptions))

for i in tqdm(range(10)): 
    # print(descriptions)
    wav = model.generate(descriptions)  # generates 3 samples.

    # melody, sr = torchaudio.load('./assets/bach.mp3')
    # generates using the melody from the given audio and the provided descriptions.
    # wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        iidx = int((idx-(idx%5))/5)
        sidx = int(idx%5)
        print(iidx, sidx)
        path = target_path/paths[iidx]
        path.mkdir(parents=True, exist_ok=True)
        audio_write(f'{str(path/caption_idx[sidx])}_{i}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        dfs[iidx].to_csv(f'{str(path)}.csv', sep = '\n')