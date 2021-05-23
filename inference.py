import sys
sys.path.append('waveglow/')
sys.path.append('tacotron2/')

import numpy as np
import torch
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from tacotron2.train import load_model
from text import text_to_sequence, sequence_to_text
from denoiser import Denoiser
import librosa
import soundfile
from hparams import create_hparams
from viphoneme import syms, vi2IPA_split
import re

hparams = create_hparams()
hparams.sampling_rate = 22050
hparams.max_decoder_steps=20000


waveglow_path = 'waveglow/waveglow_256channel.pt' # 'waveglow_256channels_new.pt' #
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

tacotron2_path = "tacotron2/checkpoint_54000"
model = load_model(hparams)
model.load_state_dict(torch.load(tacotron2_path)['state_dict'])
_ = model.cuda().eval().half()

def Taco(text,idx):

    # Normalization - Phonetization
    delimit="/"
    text=re.sub(re.compile(r'\s+'), ' ', text)
    text=text.rstrip(".").rstrip("?").rstrip("!").rstrip(" ")
    ipa = vi2IPA_split(text,delimit)
    text=ipa.replace("/'","")
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    # Synthesizer: Tacotron2
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

    # Vocoder: Waveglow
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    soundfile.write("AUDIO/tmp_taco.wav", audio[0].data.cpu().numpy().astype("float32"), 22050)

def VoiceClone(text,ref_audio):

    # Normalization - Phonetization
    delimit="/"
    text=re.sub(re.compile(r'\s+'), ' ', text)
    text=text.rstrip(".").rstrip("?").rstrip("!").rstrip(" ")
    ipa = vi2IPA_split(text,delimit)
    text=ipa.replace("/'","")
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    # Synthesizer: Tacotron2
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

    print(mel_outputs_postnet.dtype)
    # Vocoder: Waveglow
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    # Down Sample Rate
    soundfile.write("AUDIO/tmp.wav", audio[0].data.cpu().numpy().astype("float32"), 22050)
    tmp_audio, sf = librosa.load("AUDIO/tmp.wav", 16000)
    soundfile.write("AUDIO/tmp.wav", tmp_audio, sf)
    test_audio, sr = soundfile.read("AUDIO/tmp.wav")

    sys.path.append('autovc/')
    sys.path.append('speaker_verification')
    from taco_inference import generateAudio

    wave = generateAudio("AUDIO/tmp.wav" ,ref_audio,"autovc/checkpoints_wided_addnoise_final/autovc_250000.pt", "waveglow/checkpoint_step001000000_ema.pth", english=False)
    
    return wave

from MCD import evaluate_mcd_wav
if __name__ =='__main__':

    
    import glob
    ExitAudio = glob.glob("/home/trinhan/AILAB/VCSystem/AUDIO/VoiceClone/*.wav")
    ExitAudio = [e.replace("/home/trinhan/AILAB/VCSystem/AUDIO/VoiceClone/","").replace(".wav","") for e in ExitAudio ]
    print(ExitAudio)

    audio_path = "/home/trinhan/AILAB/VoiceClone/DATA/VIVOS/vivos/train/waves/"
    MCD = []
    SPEAKER = []
    with open("DATA/VoiceCloneMCDtrain.txt", "r",encoding="utf-8") as f:
        lines = f.read().splitlines()
        for idx,text in enumerate(lines[1:]):
            
            if str(idx) in ExitAudio:
                print("Audio was generated: ",idx)

                script , ref , ground = text.split("\t")
                SPEAKER.append(ref.split("_")[0])
                
                continue

            script , ref , ground = text.split("\t")

            if ref.split("_")[0] in SPEAKER:
                continue
            
            speaker_ref, _ = ref.split("_")
            speaker_ground, _ = ground.split("_")

            ref_path = audio_path + speaker_ref + "/" + ref + ".wav"
            ground_path = audio_path + speaker_ground + "/" + ground + ".wav"

            print(ref_path)

            #try:
            #print("Generate audio: ",idx)
            wave = VoiceClone(script,ref_path)
            #except:
            #    print("Audio return null due to verification module")
            #    continue

            if wave is None:
                print("Audio return null due to verification module")
                continue
            
            print(wave)
            soundfile.write("AUDIO/VoiceClone/"+str(idx)+".wav", wave, 16000)
        
            mcd = evaluate_mcd_wav(ground_path,"AUDIO/VoiceClone/"+str(idx)+".wav")
            MCD.append(mcd)
            
            print(str(idx)+" - "+script)
            print(sum(MCD)/len(MCD))

            SPEAKER.append(ref.split("_")[0])
