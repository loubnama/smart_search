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
import uvicorn
import requests
import pickle
import logging
import gc  # إضافة مكتبة لتحرير الذاكرة

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Similarity Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_urls = [
    {"filename": "image1.jpg", "url": "https://drive.google.com/uc?export=download&id=1iH20YvosyOKXldMyCuUm4AKAZsWtNEOh"},
    {"filename": "image2.jpg", "url": "https://drive.google.com/uc?export=download&id=1tj3j-0dGs6V7RMOMPRrh1Gk0jx-IHrbb"},
    {"filename": "image3.jpg", "url": "https://drive.google.com/uc?export=download&id=1rc1TWOB7GhXXrKt9808CQ85TE5pcl7_E"},
    {"filename": "image4.jpg", "url": "https://drive.google.com/uc?export=download&id=1RK8gcF7MKuwsSwOR5sgITXc6yxux85X8"},
    {"filename": "image5.jpg", "url": "https://drive.google.com/uc?export=download&id=1MBo9TAZ7usJJwK_To9u5tDY6aIbIv2cT"},
    {"filename": "image6.jpg", "url": "https://drive.google.com/uc?export=download&id=1zY5vfAtRMP-cOa35jrcUuDh0_OAiuhXs"},
    {"filename": "image7.jpg", "url": "https://drive.google.com/uc?export=download&id=1SBYeAvp2FKOXj0R_X-nYG3e_64e2L2_5"},
    {"filename": "image8.jpg", "url": "https://drive.google.com/uc?export=download&id=1U6bIsKSbGdO8Ne266I_5NJXmzvUr06T4"},
    {"filename": "image9.jpg", "url": "https://drive.google.com/uc?export=download&id=1bWNcPmFUWc9HsEKjQOtHQSvwsYIv9LtP"},
    {"filename": "image10.jpg", "url": "https://drive.google.com/uc?export=download&id=10sgv7wDoWJAjmz50s_OexgnjtLVAlKUW"},
    {"filename": "image11.jpg", "url": "https://drive.google.com/uc?export=download&id=13pCAYkz-wPdrvltYrzNk4DuYuAXomIJL"},
    {"filename": "image12.jpg", "url": "https://drive.google.com/uc?export=download&id=1Cq2YuplsAHtlEX5H0aMcdDqAPZlwTvXN"},
    {"filename": "image13.jpg", "url": "https://drive.google.com/uc?export=download&id=1ZTRmokw2SA2bAhWmKbVIEPFxdAY4-6Yz"},
    {"filename": "image14.jpg", "url": "https://drive.google.com/uc?export=download&id=1HzFBzTgS11e9iV0l6tC48GOlw-ksUF65"},
    {"filename": "image15.jpg", "url": "https://drive.google.com/uc?export=download&id=1Kr9I0_qzx4uPab61670i2ckEXsIx_ZH5"},
    {"filename": "image16.jpg", "url": "https://drive.google.com/uc?export=download&id=1ya2iaHo7WOnDz6ag3cXaMgwju21C-Afu"},
    {"filename": "image17.jpg", "url": "https://drive.google.com/uc?export=download&id=15r6fV4oIO_lQE2Q9OKT_xeW_wMpwW_2X"},
    {"filename": "image18.jpg", "url": "https://drive.google.com/uc?export=download&id=13dYgzwBmfHR6ia-fFGrM9qyuGzxKHaAp"},
    {"filename": "image19.jpg", "url": "https://drive.google.com/uc?export=download&id=1YiI-N0WKwsRQRPLJDL5WJAJnQAri8q4W"},
    {"filename": "image20.jpg", "url": "https://drive.google.com/uc?export=download&id=1U6HT-Fe2f_GSMcJN_Ta-QMc9c7RrNbka"},
    {"filename": "image21.jpg", "url": "https://drive.google.com/uc?export=download&id=16Cj74mCC_MnDfMDp6K7USPfagbdibQdT"},
    {"filename": "image22.jpg", "url": "https://drive.google.com/uc?export=download&id=1pxdSSLgNYPX1Halv7ewz-NMLb6zIOBmE"},
]

image_vectors = []
image_names = []

# تحميل نموذج ResNet50
model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
model.eval()

# تحويل الصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_vector(img: Image.Image):
    img.thumbnail((256, 256))
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        vector = model(img_tensor)
    return vector.numpy().flatten()

# تحميل أو تجهيز قاعدة بيانات الصور
vectors_file = "vectors.pkl"
if os.path.exists(vectors_file):
    with open(vectors_file, "rb") as f:
        data = pickle.load(f)
        image_vectors = data["vectors"]
        image_names = data["names"]
else:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    for image_info in image_urls:
        try:
            response = requests.get(image_info["url"], headers=headers, stream=True)
            logger.info(f"Fetching {image_info['filename']}: Status {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                vec = extract_vector(img)
                image_vectors.append(vec)
                image_names.append(image_info["filename"])
                # تحرير الذاكرة
                img.close()
                del img, vec
                gc.collect()
            else:
                logger.error(f"Invalid response for {image_info['filename']}: Status {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
                if 'text/html' in response.headers.get('Content-Type', ''):
                    logger.error(f"HTML content received: {response.text[:500]}")
        except Exception as e:
            logger.error(f"Error processing image {image_info['filename']}: {str(e)}")
    if image_vectors:
        with open(vectors_file, "wb") as f:
            pickle.dump({"vectors": image_vectors, "names": image_names}, f)

@app.post("/search_by_image/")
async def search_by_image(file: UploadFile = File(...)):
    try:
        input_img = Image.open(file.file).convert("RGB")
        query_vector = extract_vector(input_img)
        input_img.close()  # تحرير الذاكرة
        del input_img
        gc.collect()

        similarities = cosine_similarity([query_vector], image_vectors)[0]
        top_indices = similarities.argsort()[-2:][::-1]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        results = []
        for idx in top_indices:
            image_info = image_urls[idx]
            try:
                response = requests.get(image_info["url"], headers=headers, stream=True)
                logger.info(f"Fetching result {image_info['filename']}: Status {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
                if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                    with Image.open(io.BytesIO(response.content)) as img:
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG", quality=50)
                        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    results.append({
                        "filename": image_names[idx],
                        "score": float(similarities[idx]),
                        "image_base64": encoded_string
                    })
                else:
                    results.append({
                        "filename": image_names[idx],
                        "score": float(similarities[idx]),
                        "error": f"Failed to load image: Status {response.status_code}, Content-Type: {response.headers.get('Content-Type')}"
                    })
            except Exception as e:
                results.append({
                    "filename": image_names[idx],
                    "score": float(similarities[idx]),
                    "error": str(e)
                })

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
