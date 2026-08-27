import io
import os
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from PIL import Image

try:
	from .dataset import get_transforms
	from .model import get_model
except ImportError:
	from dataset import get_transforms
	from model import get_model


DEFAULT_CLASSES = [
	"airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck",
]


def load_model(checkpoint_path: str, device: torch.device):
	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model = get_model(
		architecture=checkpoint.get("architecture", "cnn"),
		num_classes=checkpoint.get("num_classes", 10),
	)
	model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
	model.to(device)
	model.eval()
	return model


def create_app(checkpoint_path: str | None = None) -> Flask:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	path = checkpoint_path or os.getenv("MODEL_CHECKPOINT", "checkpoints/cifar10_cnn.pt")
	model = load_model(path, device) if Path(path).exists() else None
	app = Flask(__name__)

	@app.get("/health")
	def health():
		if model is None:
			return jsonify({"status": "unhealthy", "model_loaded": False}), 503
		return jsonify({"status": "ok", "model_loaded": True})

	@app.post("/predict")
	def predict():
		if model is None:
			return jsonify({"error": "model checkpoint is not loaded"}), 503
		image_file = request.files.get("image")
		image_bytes = image_file.read() if image_file else request.get_data()
		if not image_bytes:
			return jsonify({"error": "provide an image multipart field named 'image' or a raw image body"}), 400
		try:
			image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
		except (OSError, ValueError):
			return jsonify({"error": "invalid image"}), 400
		inputs = get_transforms(train=False)(image).unsqueeze(0).to(device)
		with torch.inference_mode():
			probabilities = torch.softmax(model(inputs), dim=1)[0].cpu().tolist()
		classes = DEFAULT_CLASSES[:len(probabilities)]
		return jsonify({"probabilities": probabilities, "classes": classes})

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
