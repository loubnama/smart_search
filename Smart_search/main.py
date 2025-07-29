from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import os
import base64
import io

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اسمح لجميع المصادر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# المسار للصور
images_folder = "images"
image_vectors = []
image_names = []

# تحميل نموذج التعرف على الصور
model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_vector(img: Image.Image):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        vector = model(img_tensor)
    return vector.numpy().flatten()

# تجهيز قاعدة بيانات الصور
for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(images_folder, filename)
        img = Image.open(path).convert("RGB")
        vec = extract_vector(img)
        image_vectors.append(vec)
        image_names.append(filename)

@app.post("/search_by_image/")
async def search_by_image(file: UploadFile = File(...)):
    try:
        # فتح الصورة المدخلة
        input_img = Image.open(file.file).convert("RGB")
        query_vector = extract_vector(input_img)

        # حساب التشابه
        similarities = cosine_similarity([query_vector], image_vectors)[0]
        top_indices = similarities.argsort()[-5:][::-1]

        # تجهيز الصور المشابهة
        results = []
        for idx in top_indices:
            image_path = os.path.join(images_folder, image_names[idx])
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            results.append({
                "filename": image_names[idx],
                "score": float(similarities[idx]),
                "image_base64": encoded_string
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
