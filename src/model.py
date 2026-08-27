import torch.nn as nn


class CIFAR10CNN(nn.Module):
	"""Small convolutional classifier for 32x32 RGB CIFAR-10 images."""

	def __init__(self, num_classes: int = 10) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Linear(128, num_classes)

	def forward(self, inputs):
		features = self.features(inputs)
		return self.classifier(features.flatten(1))


def get_model(architecture: str = "cnn", num_classes: int = 10) -> nn.Module:
	if architecture.lower() not in {"cnn", "cifar10_cnn"}:
		raise ValueError(f"Unsupported architecture: {architecture}")
	return CIFAR10CNN(num_classes=num_classes)
