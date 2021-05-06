import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from model_bl import D_VECTOR
from collections import OrderedDict


import librosa
from synthesis import build_model
from synthesis import wavegen

from make_spect import makeSpect

import soundfile as sf

import sys
sys.path.append('./speaker_verification/')


from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk

encoder = SpeechEmbedder()
encoder.load_state_dict(torch.load("speaker_verification/final_epoch_950_batch_id_103.model"))
encoder.eval()

def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
    return concat_seg

def get_STFTs(segs):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=hp.data.nfft,
                            win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr),pad_mode='empty')
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
            else:
                break
    return STFT_frames

def align_embeddings(embeddings):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    avg_embeddings = np.zeros((len(partitions),256))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0)
    return avg_embeddings

def get_embedding(audio_path):
    times, segs = VAD_chunk(2, audio_path)
    #print("segs ", segs.shape)
    if segs == []:
        print('No voice activity detected')
        return None
    concat_seg = concat_segs(times, segs)
    STFT_frames = get_STFTs(concat_seg)
    STFT_frames = np.stack(STFT_frames, axis=2)
    STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)))
    #print("STFT shape: ", STFT_frames.shape)
    embeddings = encoder(STFT_frames)
    return embeddings

def get_verification_pytorch(audio_path):
    embed1 = get_embedding(audio_path)
    if embed1 is None: return None
    embed1 = align_embeddings(embed1.detach().numpy()) #encoder.embed_utterance(wav1)
    embed1 = np.mean(embed1, axis=0)
    return embed1

### AUDIO AND CHECKPOINT PATH
#ref_audio = "vn_wavs/audioclip-1615615975000-3298.wav" # first audio
#original_audio = "vn_wavs/bo_gia.wav" # second audio
#original_audio = "vn_wavs/1/5.wav"
#ref_audio = "vn_wavs/2/6.wav"

#ref_audio = "vn_wavs/database_sa2_Feb29_Mar06_Mar17_toApril06_cleaned_utt_0000229798-1.wav"
#original_audio = "vn_wavs/database_sa2_Feb29_Mar06_Mar17_toApril06_cleaned_utt_0000153098-1.wav"
ref_audio = "../vivos_only_wavs/VIVOSSPK24/VIVOSSPK24_001.wav" 
original_audio = "../vivos_only_wavs/VIVOSSPK14/VIVOSSPK14_001.wav"

autovc_checkpoint = 'checkpoints_fully/autovc_550000.pt'
speaker_encoder_checkpoint = "../3000000-BL.ckpt"
###

name_or = original_audio.split("/")[-1][:-4]
name_ref = ref_audio.split("/")[-1][:-4]


### GENERATE MEL
mel_org = makeSpect(original_audio, None)
mel_ref = makeSpect(ref_audio, None)
###

### GENERATE SPEAKER EMBED
#C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
#c_checkpoint = torch.load(speaker_encoder_checkpoint)
#new_state_dict = OrderedDict()
#for key, val in c_checkpoint['model_b'].items():
#    new_key = key[7:]
#    new_state_dict[new_key] = val
#C.load_state_dict(new_state_dict)

#emb_org = C(torch.FloatTensor(mel_org[np.newaxis, :, :]).cuda())
#emb_ref = C(torch.FloatTensor(mel_ref[np.newaxis, :, :]).cuda())
###

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'
G = Generator(32,256,512,32).eval().to(device)

g_checkpoint = torch.load(autovc_checkpoint, map_location=torch.device('cuda'))

#G.load_state_dict(g_checkpoint['model'])
G = g_checkpoint

x_org = mel_org
x_org, len_pad = pad_seq(x_org)
uttr_org = torch.FloatTensor(x_org[np.newaxis, :, :]).to(device)

emb_org = get_verification_pytorch(original_audio)
emb_ref = get_verification_pytorch(ref_audio)

'''
if emb_org is None or emb_ref is None:
    print(emb_org)
    print(emb_ref)
    exit(0)

if np.isnan(np.sum(emb_org)) or np.isnan(np.sum(emb_ref)): 
    print(emb_org)
    print(emb_ref)
    exit(0)
'''

emb_org = torch.FloatTensor(emb_org).unsqueeze(0).cuda()
emb_ref = torch.FloatTensor(emb_ref).unsqueeze(0).cuda()



#org_idx, ref_idx = 14,19

#emb1,emb2 = np.zeros(65,dtype="float32"),np.zeros(65,dtype="float32")
#emb1[org_idx-1],emb2[ref_idx-1]=1,1

#emb_org = torch.FloatTensor(emb1).unsqueeze(0).cuda()
#emb_ref = torch.FloatTensor(emb2).unsqueeze(0).cuda()


with torch.no_grad():
    _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_ref)
    
if len_pad == 0:
    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
else:
    uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()


device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("../checkpoint_step001000000_ema.pth",map_location=torch.device('cuda'))
model.load_state_dict(checkpoint["state_dict"])

#utter_trg = np.load("../database_sa2_Feb29_Mar06_Mar17_toApril06_cleaned_utt_0000153098-1.npy")
waveform = wavegen(model, c=uttr_trg)   
#sf.write("test_22.wav", waveform, 16000,subtype='PCM_24')
sf.write('{}-{}.wav'.format(name_or, name_ref), waveform, 16000,subtype='PCM_24')
