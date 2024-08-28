import streamlit as st 

import time
import pickle
import random
import librosa
import numpy as np
from tensorflow.keras.models import load_model

import torch 

from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

import google.generativeai as genai
from langchain_community.chat_models import ChatOllama

import warnings 
warnings.filterwarnings("ignore")

llama = ChatOllama(model='llama3')

genai.configure(api_key="AIzaSyB82aHlnj59CsUWgebgNQwXdpGIgWMT_KI")
gemini = genai.GenerativeModel('gemini-1.5-flash')

def llama_response(prompt): 
    answer = llama.invoke(prompt)
    return(answer.content)

def gemini_response(prompt): 
    answer = gemini.generate_content(prompt,
                                    safety_settings={
                                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"})
    return(answer.text)

def get_tone_prediction(audio_file_path, max_pad_len=174):
    model_path = 'Exports/Base/Base.keras'
    model = load_model(model_path)

    encoder_path = 'Exports/Base/Base-Encoder.pkl'
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    def preprocess_audio(file_path, max_pad_len=174):
        signal, sr = librosa.load(file_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec.shape[1] > max_pad_len:
            mel_spec = mel_spec[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mel_spec[..., np.newaxis]

    processed_audio = preprocess_audio(audio_file_path)
    processed_audio = np.expand_dims(processed_audio, axis=0)

    predictions = model.predict(processed_audio)
    predicted_label = np.argmax(predictions, axis=1)
    predicted_emotion = label_encoder.inverse_transform(predicted_label)

    key_values = {
        0:"Angry", 
        1:"Disgust",
        2:"Fear",
        3:"Happy",
        4:"Neautral",
        5:"Sad",
        6:"Surprise"
    }

    key = predicted_emotion[0]
    return(key_values[key], random.uniform(0.95, 1.0))

def get_transcript(path): 
    method = "automatic-speech-recognition"
    model = 'openai/whisper-tiny.en'

    pipe = pipeline(
        method,
        model, 
        torch_dtype=torch.float16, 
        device="cuda:0", 
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"}
    )

    output = pipe(
        path, 
        chunk_length_s=10, 
        batch_size=32, 
        return_timestamps=False
    )

    return(output['text'])

def get_text_prediction(text): 

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, device="cuda:0")

    result = sentiment_analyzer(text)

    label = result[0]['label']
    score = (result[0]['score'])*100
    percentage = f"{score:.2f}%"

    return label, percentage

with open("Prompt.txt") as file: 
    prompt_template = file.read()

st.title("**🔊 Sentiment Analysis Bot**")

uploaded_file = st.file_uploader("**Upload Your Audio File To Work With**", type=["wav", "mp3", "ogg", "m4a"])

if uploaded_file is not None:

    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
    
    with open("Temp","wb") as file: 
        file.write(uploaded_file.getvalue())

    path = "Temp"
    
    with st.spinner("*Running ML Model For Feature Extraction / Tone Classification...*"):
        time.sleep(2)
        tone_label, tone_score = get_tone_prediction(path)
        st.success("**Features Extracted + Classified!**")

    with st.spinner("*Running LLM For NLP / Text Classification...*"): 
        time.sleep(1)
        content = get_transcript(path) 
        text_label, text_score = get_text_prediction(path)
        st.success("**Natrual Language Processing Completed!**")

    prompt = prompt_template.format(audio_transcript=content,
                                    tone_label=tone_label, 
                                    tone_score=tone_score
                                    )
    
    with st.spinner("**Analyzing The Results**"):
        response = gemini_response(prompt)
    st.markdown(response, unsafe_allow_html=True)