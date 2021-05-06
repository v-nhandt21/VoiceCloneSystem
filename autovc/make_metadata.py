"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
#import speaker_dct from speaker_dct

import sys
sys.path.append('./speaker_verification/')

#################### 
#from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from numpy import dot
from numpy.linalg import norm
import random

#
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk
#

import torch
import librosa
import math

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
    #if embed1 == None: return None
    embed1 = align_embeddings(embed1.detach().numpy()) #encoder.embed_utterance(wav1)
    embed1 = np.mean(embed1, axis=0)
    return embed1
################################






#C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
#c_checkpoint = torch.load('../3000000-BL.ckpt')
#new_state_dict = OrderedDict()
#for key, val in c_checkpoint['model_b'].items():
#    new_key = key[7:]
#    new_state_dict[new_key] = val
#C.load_state_dict(new_state_dict)
num_uttrs = 20
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    idxs = int(speaker[-2:])
    print('IDX: %s' % str(idxs))

    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding

    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        
        #print(i,"==utter")
        #tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        #candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short

        #while tmp.shape[0] < len_crop:
        #    idx_alt = np.random.choice(candidates)
        #    tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
        #    candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        #left = np.random.randint(0, tmp.shape[0]-len_crop)
        #melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        
        #AUTOVC origin English
        #emb = C(melsp)
        #emb = emb.detach().squeeze().cpu().numpy()
        #embs.append(emb.detach().squeeze().cpu().numpy()) 

        #Onehot
        #emb = np.zeros(65,dtype="float32")
        #emb[idxs - 1]=1
        #embs.append(emb)

        #Verification Pytorch
        try:
            audio_path = "../vivos_only_wavs/{}/{}.wav".format(speaker,fileList[idx_uttrs[i]][:-4])
            emb = get_verification_pytorch(audio_path)

            if np.isnan(np.sum(emb)): 
                print("Have nan")
                continue

            embs.append(emb)
        except:
            continue

    assert len(embs) != 0

    utterances.append(np.mean(embs, axis=0))

    '''
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     
    '''

    #utterances.append(np.mean(embs, axis=0))
    #idx = speaker_dct[speaker]
    #num = np.array([0] * len(subdirList))
    #num[idx] = 1

    #embed = torch.HalfTensor(num)
    #utterances.append(embed)
 
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train_speaker_embed.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
