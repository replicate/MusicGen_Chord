import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.modules.conditioners import ChromaChordConditioner
from audiocraft.solvers.compression import CompressionSolver
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import torch.nn as nn
import os 
import math
import torch
from omegaconf import OmegaConf
from audiocraft.models.builders import get_lm_model

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

device = 'cuda'

loaded = torch.load("/home/sake/musicgen_chord.th")
def _delete_param(cfg, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)

# LM

cfg = OmegaConf.create(loaded['xp.cfg'])
cfg.device = str(device)
if cfg.device == 'cpu':
    cfg.dtype = 'float32'
else:
    cfg.dtype = 'float16'
_delete_param(cfg, 'conditioners.self_wav.chroma_chord.cache_path')
_delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
_delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
_delete_param(cfg, 'conditioners.args.drop_desc_p')

lm = get_lm_model(loaded['xp.cfg'])
lm.load_state_dict(loaded['best_state']['model']) 
lm.eval()
lm.cfg = cfg

compression_model = CompressionSolver.wrapped_model_from_checkpoint(cfg, cfg.compression_model_checkpoint, device=device)

model = MusicGen("sakemin/musicgen-chord", compression_model, lm)

model.set_generation_params(duration=30)
model.lm = model.lm.to(device)

target_path = Path("/home/sake/chords_text_gen")

# caption_idx = ["tag", "caption_writing", "caption_summary", "caption_paraphrase", "caption_attribute_prediction"]

# rand_idxs = random.sample(range(1, len(df)), 4)

descriptions = ["bossa nova", "bossa nova"]
chord_text = ['C G A:min F', 'F G E:min A:min']
bpm = 60
in_triple = False

for i in range(len(descriptions)):
    if in_triple:
        descriptions[i] = descriptions[i] + ", in triple"
    descriptions[i] = descriptions[i] + f", bpm : {bpm}"

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
for i in tqdm(range(1)): 
    # print(descriptions)
    # wav = model.generate(descriptions)  # generates 3 samples.

    # melody, sr = torchaudio.load('/home/sake/01 Psycho.mp3')
    # wav = model.generate_with_chroma(descriptions, melody[None], sr)

    wav = model.generate_with_text_chroma(descriptions, chord_text, bpm = bpm, in_triple = in_triple)

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        # iidx = int((idx-(idx%5))/5)
        # sidx = int(idx%5)
        # print(iidx, sidx)
        # path = target_path/paths[iidx]
        path.mkdir(parents=True, exist_ok=True)
        audio_write(f'{str(path/str(idx))}_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)