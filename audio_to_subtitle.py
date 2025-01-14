from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import torchaudio
import librosa
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.inference.interfaces import foreign_class
import numpy as np


# モデルのロード
def load_asr_model():
    return EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
    )

def load_emotion_model():
    return foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
    )

# 音声認識の実行
def transcribe_audio_with_timestamps(audio_file, asr_model):
    speech, sample_rate = torchaudio.load(audio_file)
    duration = len(speech[0]) / sample_rate
    transcript = asr_model.transcribe_file(audio_file)
    words = transcript.split()
    word_count = len(words)
    avg_word_duration = duration / word_count
    transcription = []
    current_time = 0.0

    for word in words:
        start_time = round(current_time, 2)
        end_time = round(start_time + avg_word_duration, 2)
        transcription.append({
            "word": word,
            "start_time": start_time,
            "end_time": end_time,
        })
        current_time = end_time

    return transcription

# 音声のRMSを単語ごとに計算
def compute_rms_per_word(audio_file, transcription):
    y, sr = librosa.load(audio_file, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)
    word_rms = []

    for word in transcription:
        start_time, end_time = word["start_time"], word["end_time"]
        rms_values_in_word = [
            rms[i] for i, t in enumerate(times) if start_time <= t <= end_time
        ]
        avg_rms = np.mean(rms_values_in_word) if rms_values_in_word else 0.1  # デフォルト値
        word_rms.append(avg_rms)

    return word_rms

# 感情分類
def classify_emotion(audio_file, emotion_model):
    prediction = emotion_model.classify_file(audio_file)
    return prediction

# 感情に基づくフォントカラーの取得
def get_font_color(emotion_label):
    emotion_colors = {
        "hap": "yellow",
        "ang": "red",
        "sad": "blue",
        "neu": "white"
    }
    return emotion_colors.get(emotion_label, "white")

# RMSに基づいてフォントサイズを決定
def get_font_size_from_rms(rms_value):
    font_size = int(rms_value * 5000)
    return min(max(font_size, 10), 400)

# 字幕生成
def generate_text_captions(transcription, rms_values, emotion_label):
    captions = []
    color = get_font_color(emotion_label)

    for word, rms_value in zip(transcription, rms_values):
        text = word["word"]
        start_time = word["start_time"]
        end_time = word["end_time"]
        font_size = get_font_size_from_rms(rms_value)
        captions.append((start_time, end_time, text, color, font_size))

    return captions

# 動画に字幕を追加
def add_text_captions(input_video, captions, output_video):
    video = VideoFileClip(input_video)
    clips = []

    for start, end, text, color, font_size in captions:
        duration = end - start  # 表示時間を計算
        txt_clip = TextClip(
            text,
            fontsize=font_size,
            color=color,
            bg_color="black"
        ).set_position(("center", "bottom")).set_start(start).set_duration(duration)  # 小数秒でも正確に設定
        clips.append(txt_clip)

    # 字幕を動画に合成
    video_with_captions = CompositeVideoClip([video] + clips)

    # 出力ファイルを保存
    video_with_captions.write_videofile(output_video, codec="libx264", audio_codec="aac")


# メイン処理
if __name__ == "__main__":
    input_video = "test5.mp4"
    audio_file = "test5.wav"
    output_video = "output_with_captions5.mp4"

    print("Loading ASR model...")
    asr_model = load_asr_model()

    print("Loading emotion model...")
    emotion_model = load_emotion_model()

    print("Transcribing audio...")
    transcription = transcribe_audio_with_timestamps(audio_file, asr_model)

    print("Computing RMS per word...")
    rms_values = compute_rms_per_word(audio_file, transcription)
    print(rms_values)

    print("Classifying emotion...")
    emotion_prediction = classify_emotion(audio_file, emotion_model)
    print("Emotion Prediction:", emotion_prediction)
    for label, score in zip(emotion_prediction[0], emotion_prediction[1]):
        print(f"Emotion: {label}, Score: {score}")
    emotion_label = emotion_prediction[3][0]  # 主な感情ラベルを取得
    print(f"Detected emotion: {emotion_label}")
    
    print("Generating captions...")
    captions = generate_text_captions(transcription, rms_values, emotion_label)
    print(captions)
    print("Adding captions to video...")
    add_text_captions(input_video, captions, output_video)

    print("Video processing complete. Check:", output_video)
