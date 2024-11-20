import os
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from io import BytesIO
from tensorflow.image import resize
from PIL import Image


st.set_page_config(page_title="Clasificador musica", page_icon="🤖", layout="wide")
# CSS personalizado para el fondo y estilos
@st.cache_resource()
def cargarModelo():
    ruta_modelo = os.path.join(os.getcwd(), "modelos", "Trained_model.h5")
    model = tf.keras.models.load_model(ruta_modelo)
    return model


def procesarDatos(file_stream, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_stream, sr=None)

    duracion_segmento = 4  # segundos
    duracion_superposicion = 2  # segundos

    # Convertir duraciones a muestras
    muestras_segmento = duracion_segmento * sample_rate
    muestras_superposicion = duracion_superposicion * sample_rate

    num_segmentos = int(np.ceil((len(audio_data) - muestras_segmento) / (muestras_segmento - muestras_superposicion))) + 1

    # Iterar sobre cada segmento
    for i in range(num_segmentos):
        inicio = i * (muestras_segmento - muestras_superposicion)
        fin = inicio + muestras_segmento
        segmento = audio_data[inicio:fin]
        mel_spectrograma = librosa.feature.melspectrogram(y=segmento, sr=sample_rate)
        mel_spectrograma = resize(np.expand_dims(mel_spectrograma, axis=-1), target_shape)
        data.append(mel_spectrograma)

    return np.array(data)


def predecirModelo(X_test):
    model = cargarModelo()
    y_pred = model.predict(X_test)
    predecir_Categoria = np.argmax(y_pred, axis=1)
    elementos_unicos, contador = np.unique(predecir_Categoria, return_counts=True)
    contador_maximo = np.max(contador)
    elementos = elementos_unicos[contador == contador_maximo]
    return elementos[0]


st.markdown(
    """
    <style>
    .stApp {
        background-color: #191919;
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #303030;
        color: #FFFFFF;
    }
    .stFileUploader label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.header("Clasificador de géneros musicales")

test = st.file_uploader("Subir el archivo", type=["mp3", "wav"])

if test is not None:
    # Procesar archivo desde memoria
    file_stream = BytesIO(test.read())

    # Botón para reproducir audio
    if st.button("Play Audio"):
        st.audio(file_stream)

    # Botón de predicción
    if st.button("Predecir"):
        with st.spinner("Cargando..."):
            st.snow()
            try:
                # Procesar datos desde el flujo de memoria
                X_test = procesarDatos(file_stream)
                indice = predecirModelo(X_test)
                label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                st.markdown("**El género de la música es: :red[{}]**".format(label[indice]))
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")