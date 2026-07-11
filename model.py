import os
import numpy as np
from PIL import Image, ImageOps

# Keras 3 uyumluluk yaması (Eğer eski sürüme geçmediysen hayat kurtarır)
try:
    import keras
    if hasattr(keras, 'layers') and hasattr(keras.layers, 'DepthwiseConv2D'):
        original_init = keras.layers.DepthwiseConv2D.__init__
        def patched_init(self, *args, **kwargs):
            if 'groups' in kwargs: kwargs.pop('groups')
            original_init(self, *args, **kwargs)
        keras.layers.DepthwiseConv2D.__init__ = patched_init
    from keras.models import load_model
except:
    from tensorflow.keras.models import load_model

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

model = None
labels = []

def load_teachable_machine():
    global model, labels
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("❌ HATA: Model veya etiket dosyası bulunamadı!")
        return False
    
    model = load_model(MODEL_PATH, compile=False)
    
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip().split(" ", 1)[-1].strip() for line in f if line.strip()]
    
    print("✓ MODEL BAŞARIYLA YÜKLENDİ! SİSTEM HAZIR.")
    return True

def get_class(file_path):
    if model is None:
        load_teachable_machine()
        
    image = Image.open(file_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    return labels[index], float(prediction[0][index])

# İlk çalıştırmada modeli otomatik yükle
load_teachable_machine()