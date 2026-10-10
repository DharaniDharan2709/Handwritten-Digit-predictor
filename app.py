import base64
import io

from pathlib import Path

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from torch import nn


app = FastAPI()

project_folder = Path(__file__).resolve().parent


class PredictionRequest(BaseModel):
    image: str


class DigitRecognizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(784, 30)
        self.activation = nn.Sigmoid()
        self.output_layer = nn.Linear(30, 10)

    def forward(self, images):
        x = self.flatten(images)
        z1 = self.hidden_layer(x)
        a1 = self.activation(z1)
        logits = self.output_layer(a1)

        return logits


model = DigitRecognizer()

model_file = project_folder / "digit_recognizer.pth"

saved_weights = torch.load(
    model_file,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(saved_weights)
model.eval()


@app.get("/")
def home():
    html_file = project_folder / "index.html"

    return FileResponse(html_file)


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        header, encoded_image = request.image.split(",", 1)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image data."
        )

    try:
        image_bytes = base64.b64decode(encoded_image)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the image."
        )

    image_file = io.BytesIO(image_bytes)

    try:
        image = Image.open(image_file)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the image."
        )

    grayscale_image = image.convert("L")

    bounding_box = grayscale_image.getbbox()

    if bounding_box is None:
        raise HTTPException(
            status_code=400,
            detail="Please draw a digit first."
        )

    cropped_digit = grayscale_image.crop(
        bounding_box
    )

    cropped_digit.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    processed_image = Image.new(
        mode="L",
        size=(28, 28),
        color=0
    )

    left = (
        28 - cropped_digit.width
    ) // 2

    top = (
        28 - cropped_digit.height
    ) // 2

    processed_image.paste(
        cropped_digit,
        (left, top)
    )

    image_array = np.array(
        processed_image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_tensor = torch.from_numpy(
        image_array
    )

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_digit = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_digit
        ].item()

    return {
        "digit": predicted_digit,
        "confidence": confidence
    }