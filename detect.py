import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
import numpy as np
import cv2
import face_recognition
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

from torch import nn
from torchvision import models

# --- Model Definition ---
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# --- Dataset ---
from torch.utils.data import Dataset

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=20, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        logger.info(f"Extracting frames from: {video_path}")
        for frame in self.frame_extract(video_path):
            if frame is None:
                continue
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)  # shape: [1, seq_len, 3, H, W]

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# --- LIME Explanation Wrapper ---
class LimeWrapper:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict_single(self, imgs_np):
        # imgs_np is list of images (H, W, C) uint8 in RGB
        self.model.eval()
        imgs_tensor = torch.stack([self.transform(img).to(self.device) for img in imgs_np])
        # Repeat dimension to match sequence length for LSTM input [N, seq_len, 3, 112, 112]
        imgs_tensor = imgs_tensor.unsqueeze(1).repeat(1, 20, 1, 1, 1)
        with torch.no_grad():
            _, outputs = self.model(imgs_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()

# --- Helper to convert matplotlib fig to base64 ---
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    plt.close(fig)
    return base64_str

# --- Function to generate LIME explanation image for one frame ---
def explain_frame_lime(img_np, model_wrapper):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=model_wrapper.predict_single,
        top_labels=1,
        hide_color=0,
        num_samples=10
    )
    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    lime_img = mark_boundaries(temp, mask)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(lime_img)
    ax.axis('off')
    return fig_to_base64(fig)

# --- Setup FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

device = torch.device("cpu")

model_path = 'E://project_ui//model_87_acc_20_frames_final_data.pt'

logger.info("Initializing model...")
model = Model(num_classes=2).to(device)

if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

logger.info("Loading model weights...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
logger.info("Model loaded and ready.")

lime_wrapper = LimeWrapper(model, device=device)

# --- Prediction function ---
def predict(model, img_tensor):
    logger.info("Running prediction...")
    with torch.no_grad():
        fmap, logits = model(img_tensor.to(device))
        sm = torch.nn.Softmax(dim=1)
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        logger.info(f"Prediction: {prediction.item()}, Confidence: {confidence:.2f}%")
        return int(prediction.item()), confidence

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Extract frames and prepare tensor
        video_dataset = validation_dataset([tmp_path], sequence_length=20, transform=train_transforms)
        video_tensor = video_dataset[0]  # shape [1, 20, 3, 112, 112]

        pred_label, confidence = predict(model, video_tensor)
        label_map = {0: "FAKE", 1: "REAL"}
        prediction_text = label_map.get(pred_label, "UNKNOWN")

        # Prepare frames for LIME explanation (original RGB frames)
        frames_tensor = video_tensor.squeeze(0)  # [20, 3, 112, 112]
        frames_np = []
        for ft in frames_tensor:
            # De-normalize and convert to numpy uint8 RGB
            img = ft.cpu().numpy().transpose(1,2,0)
            img = (img * np.array(std) + np.array(mean))
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            frames_np.append(img)

        # Generate LIME explanations for each frame (or subset, e.g. first 5)
        lime_images = []
        for idx, frame_np in enumerate(frames_np[:2]):  # limit to 5 frames for performance
            logger.info(f"Generating LIME explanation for frame {idx+1}")
            b64_img = explain_frame_lime(frame_np, lime_wrapper)
            lime_images.append(b64_img)

        return JSONResponse({
            "prediction": prediction_text,
            "confidence": confidence,
            "lime_explanations": lime_images
        })

    finally:
        os.remove(tmp_path)
    