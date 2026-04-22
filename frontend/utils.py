import torch
from PIL import Image
from torchvision import transforms


IMG_SIZE = 256

transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform_eval(image).unsqueeze(0)
    return tensor.to(device)


def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)

        # Binary head: output shape [B,1]
        prob_real = torch.sigmoid(logits).item()
        pred_label = "Real" if prob_real >= 0.5 else "AI-generated"
        prob_fake = 1.0 - prob_real
    return pred_label, prob_fake