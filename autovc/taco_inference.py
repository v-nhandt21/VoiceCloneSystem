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

from waveglow_vocoder import WaveGlowVocoder


encoder = SpeechEmbedder()
encoder.load_state_dict(torch.load(os.getcwd()+"/autovc/speaker_verification/final_epoch_950_batch_id_103.model"))
encoder.eval()

def isFailed(emb):
     
    try:
        if emb is None:
            return True

        if np.isnan(np.sum(emb)):
            return True
    except:
        return True
        
    return False

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

def get_verification_pytorch_1000(audio_path, count=1000):
    embed = get_verification_pytorch(audio_path)
    
    while isFailed(embed):
        embed = get_verification_pytorch(audio_path)
        count -= 1
        if count == 0: return None
    return embed

def get_verification_eng(audio_path, speaker_encoder_eng = "speaker_verification/3000000-BL.ckpt"):
    
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load(speaker_encoder_eng)
    
    #print(c_checkpoint)
    
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

    #print("go into here")
    
    mel = makeSpect(audio_path, None)
    emb = C(torch.FloatTensor(mel[np.newaxis, :, :]).cuda())
    return emb 

def new_mel_generate(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    y_tensor = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)

    mel = WV.wav2mel(y_tensor)
    return mel


###





def generateAudio(original_audio, ref_audio, autovc_checkpoint, vocoder_checkpoint ,english=False):

    mel_org = makeSpect(original_audio, None)

    def pad_seq(x, base=32):
        len_out = int(base * ceil(float(x.shape[0])/base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

    device = 'cuda:0'
    G = Generator(32,256,512,32).eval().to(device)

    g_checkpoint = torch.load(autovc_checkpoint, map_location=torch.device('cuda'))
    
    G = g_checkpoint.eval()

    x_org = mel_org
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.FloatTensor(x_org[np.newaxis, :, :]).to(device)

    emb_org = get_verification_pytorch_1000(original_audio)
    
    if not english:
        emb_ref = get_verification_pytorch_1000(ref_audio)
    else:
        emb_ref = get_verification_eng(ref_audio)
        
    if emb_org is None or emb_ref is None: return
   
    emb_org = torch.FloatTensor(emb_org).unsqueeze(0).cuda()
    if not english:
        emb_ref = torch.FloatTensor(emb_ref).unsqueeze(0).cuda()
    else:
        emb_ref = emb_ref.type(torch.cuda.FloatTensor)
    
    with torch.no_grad():
        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_ref)

    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()


    device = torch.device("cuda")
    model = build_model().to(device)
    checkpoint = torch.load(vocoder_checkpoint, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint["state_dict"])

    waveform = wavegen(model, c=uttr_trg)   
    return waveform

def generateAudioGroup(original_audio, ref_audios, autovc_checkpoint = 'checkpoints_fully/autovc_700000.pt', vocoder_checkpoint = "../checkpoint_step001000000_ema.pth"):

    mel_org = makeSpect(original_audio, None)

    def pad_seq(x, base=32):
        len_out = int(base * ceil(float(x.shape[0])/base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

    device = 'cuda:0'
    G = Generator(32,256,512,32).eval().to(device)

    g_checkpoint = torch.load(autovc_checkpoint, map_location=torch.device('cuda'))
    
    G = g_checkpoint.eval()

    x_org = mel_org
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.FloatTensor(x_org[np.newaxis, :, :]).to(device)

    emb_org = get_verification_pytorch_1000(original_audio)
    emb_refs = []
    i = 0
    
    for file in os.listdir(ref_audios):
        i += 1
        print("{}/{}".format(i, len(os.listdir(ref_audios))))
    
        emb_ref = get_verification_pytorch_1000(ref_audios + file, 1)
        if emb_ref is not None: emb_refs.append(emb_ref)
        
   
    emb_refs = np.mean(emb_refs, axis=0)
    
    emb_org = torch.FloatTensor(emb_org).unsqueeze(0).cuda()
    emb_refs = torch.FloatTensor(emb_refs).unsqueeze(0).cuda()
    
    with torch.no_grad():
        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_refs)

    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()


    device = torch.device("cuda")
    model = build_model().to(device)
    checkpoint = torch.load(vocoder_checkpoint, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint["state_dict"])

    waveform = wavegen(model, c=uttr_trg)   
    return waveform

#generateAudio("vn_wavs/1/5.wav","vn_wavs/2/6.wav", english=True)

