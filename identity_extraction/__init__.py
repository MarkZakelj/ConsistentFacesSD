from time import perf_counter

import clip
import torch
from PIL import Image


def main():
    device = "cuda" if torch.cuda.is_available() else "mps"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    image2 = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat", "a machine learning"]).to(
        device
    )
    print(text.shape)

    with torch.no_grad():
        t0 = perf_counter()
        print(image.shape)
        image_features = model.encode_image(image)
        t1 = perf_counter()
        text_features = model.encode_text(text)
        t2 = perf_counter()
        print(f"Image features: {t1 - t0}, Text features: {t2 - t1}")

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs_text = logits_per_text.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


if __name__ == "__main__":
    main()
