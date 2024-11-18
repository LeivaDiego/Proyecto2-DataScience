import streamlit as st
import os
import glob
import re
import numpy as np
import pandas as pd
import torch
import cv2
import pydicom
from typing import List
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


# Configuración de dispositivo
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1  # Batch pequeño para procesamiento de un solo paciente

print(DEVICE)

# Configuración de modelos y paths
WEIGHTS = T.Compose([T.ToTensor(), T.Resize((224, 224))])  # Reemplaza con WEIGHTS.transforms() si es necesario
EFFNET_CHECKPOINTS_PATH = "models"  # Reemplaza con la ruta correcta

# Lista de nombres de modelos
MODEL_NAMES = [f'effnetv2-f{i}' for i in range(5)]
FRAC_COLS = [f'C{i}_effnet_frac' for i in range(1, 8)]
VERT_COLS = [f'C{i}_effnet_vert' for i in range(1, 8)]
columns_to_transform = ['patient_overall'] + [f'C{i}' for i in range(1, 8)]

# Función para cargar modelos
def load_model(model, name, path='.') -> torch.nn.Module:
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE, weights_only=True)
    model.load_state_dict(data)
    return model

# Cargar imagen DICOM
def load_dicom(path):
    try:
        img = pydicom.dcmread(path)
        data = img.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img
    except pydicom.errors.InvalidDicomError:
        print(f"Error loading DICOM file: {path}")
        return None, None

# Dataset personalizado para EfficientNet
class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path):
        super().__init__()
        self.df = df
        self.path = path
        
    def __getitem__(self, i):
        path = os.path.join(self.path, f'{self.df.iloc[i].Slice}.dcm')
        img, _ = load_dicom(path)
        if img is None:
            return torch.zeros(3, 224, 224)
        
        # Preprocesar la imagen
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))  # Convertir a (canales, altura, ancho)
        img = img.astype(np.float32) / 255.0  # Normalizar los valores de píxeles a [0, 1]
        return torch.from_numpy(img)
    
    def __len__(self):
        return len(self.df)

# Definición del modelo EfficientNet para predicción
class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = efficientnet_v2_s()
        self.model = create_feature_extractor(effnet, {'flatten': 'flatten'})
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 7)
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7)
        )

    def forward(self, x):
        x = self.model(x)['flatten']
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)
    

def create_probability_plot(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    regions = data["index"]
    probabilities = data[0]
    complement = 1 - probabilities

    bar_width = 0.5
    x_pos = range(len(regions))

    # Create red and green sections
    ax.bar(x_pos, probabilities, color="red", label="Probabilidad de Fractua", width=bar_width)
    ax.bar(x_pos, complement, bottom=probabilities, color="green", label="Probabilidad de Estar Sano", width=bar_width)

    # Configure plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions, rotation=45, ha="right")  # Rotate labels for better readability
    ax.set_ylabel("Value")
    ax.set_title("Probabilities and Complements")
    ax.legend(loc="upper right")

    return fig


# Predicción usando modelos EfficientNet
def predict_effnet(models: List[EffnetModel], ds) -> np.ndarray:
    dl_test = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for X in dl_test:
            pred = torch.zeros(len(X), 14).to(DEVICE)
            for m in models:
                y1, y2 = m.predict(X.to(DEVICE))
                pred += torch.cat([y1, y2], dim=1) / len(models)
            predictions.append(pred)
    return torch.cat(predictions).cpu().numpy()

# Calcular predicción final del paciente
def patient_prediction(df):
    c1c7 = np.average(df[FRAC_COLS].values, axis=0, weights=df[VERT_COLS].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7)
    return pd.Series(data=np.concatenate([[pred_patient_overall], c1c7]), index=['patient_overall'] + [f'C{i}' for i in range(1, 8)])

# Predicción para un paciente individual
def predict_single_patient(models: List[EffnetModel], patient_path: str, threshold: float):
    dicom_files = glob.glob(f'{patient_path}/*.dcm')
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {patient_path}")
    
    slices = [(os.path.basename(patient_path), int(re.search(r'(\d+)\.dcm', f).group(1))) for f in dicom_files]
    df_patient_slices = pd.DataFrame(slices, columns=['StudyInstanceUID', 'Slice']).sort_values('Slice')

    ds_patient = EffnetDataSet(df_patient_slices, patient_path)
    effnet_pred = predict_effnet(models, ds_patient)

    df_effnet_pred = pd.DataFrame(data=effnet_pred, columns=FRAC_COLS + VERT_COLS)
    df_patient_pred = pd.concat([df_patient_slices, df_effnet_pred], axis=1)
    pred_final = patient_prediction(df_patient_pred)

    # Aplicar umbral
    # pred_final[columns_to_transform] = pred_final[columns_to_transform].apply(lambda x: 1 if x > threshold else 0)
    return pred_final

# Función para crear un GIF a partir de imágenes DICOM
def create_gif_from_dicom(patient_path):
    dicom_files = sorted(glob.glob(f'{patient_path}/*.dcm'), key=lambda f: int(re.search(r'(\d+)\.dcm', f).group(1)))
    images = []
    for dicom_file in dicom_files:
        img, _ = load_dicom(dicom_file)
        if img is not None:
            images.append(Image.fromarray(img))  # Convertir a formato PIL Image
    gif_path = f"{patient_path}/animated.gif"
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=500)  # Animación de 500ms entre frames
    return gif_path

# Configuración de Streamlit
st.set_page_config(
    page_title="Predicción de Fracturas",
    page_icon="🦴",  # Favicon relacionado con huesos
    layout="centered",
    initial_sidebar_state="expanded"
)

# Paleta de colores de medicina
st.markdown(""" 
    <style>
        .css-1d391kg {
            background-color: #4CAF50;  /* Verde medicinal */
            color: white;
        }
        .css-15tx938 {
            background-color: #B2DFDB;  /* Azul claro */
        }
        .css-13y3dvo {
            background-color: #f0f0f0;  /* Fondo gris claro */
        }
        .css-1c6nxl0 {
            font-size: 18px;
            line-height: 1.6;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("Predicción de Fracturas En Vértebras")


# Cargar modelos
with st.spinner("Cargando..."):
    progress_bar = st.progress(0)
    effnet_models = []
    for idx, name in enumerate(MODEL_NAMES):
        model = load_model(EffnetModel(), name, EFFNET_CHECKPOINTS_PATH).to(DEVICE)
        effnet_models.append(model)
        progress_bar.progress((idx + 1) / len(MODEL_NAMES))  # Actualiza la barra de progreso

# Definir el directorio del paciente
patient_path = st.text_input("Introduce la ruta de las imágenes DICOM del paciente:", 'data/train_images/1.2.826.0.1.3680043.14')

# Agregar la barra deslizante para el umbral
# threshold = st.slider("Selecciona el umbral para la predicción:", 0.0, 1.0, 0.5)
# Explicación sobre el umbral de predicción
st.markdown("""    
    A continuación se desplegará una gráfica que representa las probabilidades de tener una vértebra rota. Adicionalmente se muestra una visualización 
    del caso y las probabilidades específicas de fractura para cada vértebra.
""")

# Botón de predicción
# Generate plot and display in Streamlit
if st.button("Predecir"):
    if patient_path:
        if os.path.exists(patient_path):
            try:
                pred_result = predict_single_patient(effnet_models, patient_path, threshold)

                st.write("### Predicciones Finales del Paciente:")
                st.write(pred_result)

                # Prepare data for plotting
                pred_result = pred_result.T.reset_index()
                # pred_result.columns = ["Region", "Probability"]

                # Create and display the new plot
                fig = create_probability_plot(pred_result)
                st.pyplot(fig)

                # Create GIF of the DICOM images
                gif_path = create_gif_from_dicom(patient_path)
                st.image(gif_path, caption="Animación de las imágenes DICOM", use_container_width=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")                
        else:
            st.error("La ruta del paciente no existe.")