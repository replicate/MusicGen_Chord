import torchaudio
import torch
from audiocraft.models import MusicGen
from audiocraft.solvers import MusicGenSolver
from audiocraft.data.audio import audio_write
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import torch.nn as nn
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'


solver = MusicGenSolver.get_eval_solver_from_sig('12abe086')
solver.model.cfg = solver.cfg
musicgen = MusicGen(name='mymusicgen', compression_model=solver.compression_model, lm=solver.model)
musicgen.set_generation_params(duration=30)

target_path = Path("/home/sake/audiocraft_chuki_medium")


descriptions = ["deep techno in style of chukimaandal, deep techno, dark deep techno, bpm: 133"]

for i in tqdm(range(10)): 
    # print(descriptions)
    wav = musicgen.generate(descriptions)  # generates 3 samples.
    audio_write(f'{str(target_path)}/chuki_{i}', wav[0].cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)
