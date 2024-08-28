# Advanced Sentiment Analysis: Audio & Text Integration 🎙️📜

## 🌟 Project Overview

Welcome to the **Advanced Sentiment Analysis** project! This cutting-edge system combines advanced audio processing and Natural Language Processing (NLP) to evaluate sentiments expressed in audio recordings. By integrating audio feature extraction with NLP, our system provides a holistic analysis of both the emotional tone of the speaker and the sentiment of the transcribed text. We meticulously compare these analyses to identify inconsistencies and validate the authenticity of expressed emotions.

## 🚀 How It Works

1. **🔊 Audio Input**: Upload your audio file to initiate the analysis.
2. **🛠️ Feature Extraction**: Extract crucial acoustic features from the audio:
   - **🎵 Mel-Frequency Cepstral Coefficients (MFCCs)**: Captures spectral features of the audio.
   - **🎹 Chroma Features**: Analyzes pitch class distribution.
   - **🎚️ Spectral Contrast**: Measures amplitude differences in the sound spectrum.
   - **📉 Zero-Crossing Rate**: Assesses the noisiness of the audio signal.
   - **⏱️ Temporal Features**: Includes energy and pitch variations.
3. **🔍 Tone Classification**: Classify the emotional tone into:
   - **😡 Angry**
   - **😀 Happy**
   - **😢 Sad**
   - **😐 Neutral**
   - **🔢 Many More**
4. **✍️ Transcript Generation**: Convert audio to text using automatic speech recognition (ASR).
5. **🧠 NLP Sentiment Analysis**: Analyze the transcript with advanced NLP techniques:
   - **🙂 Positive**
   - **😞 Negative**
   - **😐 Neutral**
6. **⚖️ Consistency Validation**: Compare the audio tone and text sentiment using a Large Language Model (LLM) to identify discrepancies and assess authenticity.

## 🛠️ Key Features

- **🎶 Comprehensive Audio Feature Extraction**: Captures detailed acoustic metrics for accurate emotional tone analysis.
- **🤖 Robust Tone Classification**: Employs machine learning to categorize emotions with high precision.
- **📝 Advanced Transcript Generation**: Utilizes state-of-the-art ASR technology for accurate text conversion.
- **📊 Sophisticated NLP Sentiment Analysis**: Leverages cutting-edge NLP models for nuanced sentiment classification.
- **🔬 Integrated Consistency Check**: Uses LLMs to cross-verify and validate emotional consistency between audio and text.

## 🔧 Installation Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SiddarthAA/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Setup the Environment**:
   - Create a virtual environment:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - **Windows**:
       ```bash
       venv\Scripts\activate
       ```
     - **macOS/Linux**:
       ```bash
       source venv/bin/activate
       ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**:
   - Download the necessary pre-trained models from the link below and place them in the `Exports/Base` directory.
   - [Download Models](#) 🗂️

## 🚀 Running the Application

To interact with the application via the Streamlit UI, use the following command:
```bash
streamlit run App.py
```

## 🤝 Contributing

We welcome contributions to enhance this project. Please follow these guidelines:
- Open an issue to discuss major changes.
- Submit pull requests for improvements and bug fixes.

## 📜 License
**No License**

## 📬 Contact

For inquiries, feedback, or further information, please reach out to [siddartha_ay@protonmail.com](mailto:siddartha_ay@protonmail.com).

---

Dive into the repository, explore the advanced features, and contribute to the next frontier of sentiment analysis! 🎉🌍
Kuddos :D!
