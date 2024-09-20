import streamlit as st
import time
import whisper

from pydub import AudioSegment


st.title("Audio transcription system")
st.sidebar.title("Model fine-tuning")
model = st.sidebar.selectbox("Select the whisper model that you want to use", ('tiny', 'base', 'small', 'medium', 'large'))

whisper_model = whisper.load_model(model)

def transcribe_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        audio.export("temp.wav", format="wav")
        result = whisper_model.transcribe("temp.wav")
        return result['text']
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

if "transcription" not in st.session_state:
    st.session_state.transcription = ""

audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "flac"])

if audio_file is not None:
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio("uploaded_audio.wav", format="audio/wav")

    with st.spinner('Transcribing audio...'):
        start = time.process_time()
        transcription = transcribe_audio("uploaded_audio.wav")
        end = time.process_time()

        if transcription:
            st.session_state.transcription = transcription
            st.write(f"Transcription completed in {end - start:.2f} seconds")
            st.write(transcription)
        else:
            st.write("Transcription failed. Please try again.")
