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

def VoiceClone(text,ref,idx):

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
    #mel_outputs_postnet = mel_outputs_postnet.cuda().float()

    print(mel_outputs_postnet.dtype)
    # Vocoder: Waveglow
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    

    soundfile.write("AUDIO/tmp.wav", audio[0].data.cpu().numpy().astype("float32"), 22050)
    
    # Down Sample Rate
    tmp_audio, sf = librosa.load("AUDIO/tmp.wav", 16000)
    soundfile.write("AUDIO/tmp.wav", tmp_audio, sf)
    test_audio, sr = soundfile.read("AUDIO/tmp.wav")

    sys.path.append('autovc/')
    sys.path.append('autovc/speaker_verification')
    from taco_inference import generateAudio

    wave = generateAudio("AUDIO/tmp.wav" ,ref,"autovc/checkpoints_wided_addnoise_final/autovc_250000.pt", "waveglow/checkpoint_step001000000_ema.pth", english=False)

    librosa.output.write_wav("AUDIO/VoiceClone/"+str(idx)+".wav", wave, 16000)

if __name__ =='__main__':
    texts = ["ở bất kỳ tình huống nào , cũng không được để xảy ra việc bạo hành trẻ",
    "á hậu hoàn vũ hoàng thuỳ , và người mẫu cao ngân . chúc mừng hương giang",
    "oanh cũng không bị áp lực , khi ngồi ghế nóng , giám khảo . trong chương trình , gương mặt thân quen",
    "ẩn phía sau các bề mặt phẳng , của tàu , là những hệ thống vũ khí tối tân",
    "ẩm thực và đồ nội thất , là hai mảng kinh doanh nổi bật . của đại gia đình anh",
    "cả ba đều đang kiên nhẫn chờ đợi cơ hội , đến với cuộc đời",
    "ấn độ không còn là nền kinh tế lớn phát triển nhanh nhất thế giới",
    "sách điện tử , như người bạn mới , sẽ dần dần trở thành quen thuộc , với mọi người",
    "ai cần thì gọi vào số máy ghi sẵn trên tờ rơi , dán ở các cột điện",
    "ước mơ được định cư hợp pháp , và đi học đại học ở mỹ . tan theo mây khói",
    "ốc lượm về trước tiên , cho vào thau ngâm với nước vo gạo . để ốc nhả bớt chất bẩn",
    "pha chế , là nghề được nhiều bạn theo đăng ký , trong đầu năm . tại trung cấp việt giao",
    "gạch không nung , có nhiều ưu điểm vượt trội , hơn gạch nung thông thường",
    "ít ai biết , mối tình đầu lãng mạn của tôi . là một chàng trai việt đấy",
    "da của bạn không được đẹp . thì cũng không cần lo lắng nhiều . mà chỉ cần sử dụng mật ong",
    "ủng hộ đồng bào các tỉnh miền núi phía bắc . bị thiệt hại , do thiên tai",
    "bà ấy được quyết định , là phải tránh xa các nhiệm vụ trong quyền hành của bà",
    "oanh được anh dũng đón trở về nhà . khi những ngày tết đã cận kề",
    "ước gì tôi được nhảy vào một cỗ máy thời gian . và mang tất cả trở lại",
    "bà ấy không viết . và rồi họ không tín nhiệm . để bà ấy , làm chủ tịch hội phụ nữ xã nữa"]

    for idx,text in enumerate(texts):
        ref = "../VoiceConversion/vivos_only_wavs/VIVOSSPK24/VIVOSSPK24_001.wav"
        VoiceClone(text,ref,idx)