"""Listing photo embeddings via a frozen pretrained CNN. Opt-in: requires
`torch`/`torchvision` and network access to download photos (listings.csv only
stores `picture_url`, not the image bytes).
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
import requests


def download_image(url, timeout=8):
    """Fetch a single photo and return it as a PIL Image (RGB), or None on failure."""
    from PIL import Image

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def download_images_concurrent(urls, max_workers=32, timeout=8):
    """Download many photos in parallel. Returns {url: PIL.Image} for successful downloads."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(lambda u: download_image(u, timeout=timeout), urls))
    return {url: img for url, img in zip(urls, images) if img is not None}


def build_image_embeddings(images_by_id, batch_size=32):
    """Run a frozen, ImageNet-pretrained ResNet18 over each image and return a
    dataframe of 512-d embeddings (columns 'img_0'..'img_511') indexed by id.
    """
    import torch
    from torchvision import models, transforms

    ids = list(images_by_id.keys())
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # drop the classification head, keep the 512-d pooled features
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            batch = torch.stack([preprocess(images_by_id[i]) for i in batch_ids])
            embeddings.append(model(batch).numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    columns = [f"img_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=columns, index=pd.Index(ids, name="id"))
