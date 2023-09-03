import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.modules.conditioners import ChromaChordConditioner
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import torch.nn as nn
import os 
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

model = MusicGen.get_pretrained('facebook/musicgen-melody', device='cuda')
model.set_generation_params(duration=30)
chordprovider = model.lm.condition_provider.conditioners.self_wav
model.lm.condition_provider.conditioners.self_wav = ChromaChordConditioner(chordprovider.output_dim, chordprovider.sample_rate, chordprovider.chroma.n_chroma, int(math.log2(chordprovider.chroma.winlen)), chordprovider.duration, device='cuda')
# model.to('cuda')
# model.device='cuda:1'
model.lm.to('cuda')
# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples

target_path = Path("/home/sake/chords_gen")

# caption_idx = ["tag", "caption_writing", "caption_summary", "caption_paraphrase", "caption_attribute_prediction"]

# rand_idxs = random.sample(range(1, len(df)), 4)

descriptions = ["k pop, girl group, 2022"]
'''
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
'''
# print(len(descriptions))

path = target_path/descriptions[0]
for i in tqdm(range(20)): 
    # print(descriptions)
    # wav = model.generate(descriptions)  # generates 3 samples.

    melody, sr = torchaudio.load('/home/sake/psycho_separated_drumless/mdx_extra/mdx_extra/psycho/no_drums.wav')
    # generates using the melody from the given audio and the provided descriptions.
    wav = model.generate_with_chroma(descriptions, melody[None], sr)

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        # iidx = int((idx-(idx%5))/5)
        # sidx = int(idx%5)
        # print(iidx, sidx)
        # path = target_path/paths[iidx]
        path.mkdir(parents=True, exist_ok=True)
        audio_write(f'{str(path/str(i))}_{i}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)