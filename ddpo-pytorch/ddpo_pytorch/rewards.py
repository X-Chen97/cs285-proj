from PIL import Image
import io
import numpy as np
import torch

import seaborn as sns
import pandas as pd
import os

import glob
import matplotlib.pyplot as plt

import transformers
from transformers import pipeline

# Function to classify an image and return the classification
def classify_image(image, pipe, THRESHOLD):
    result = pipe(image)
    male_score = next((item['score'] for item in result if item['label'] == 'male'), None)
    female_score = 1 - male_score if male_score is not None else None
    
    # Apply threshold and return classification
    if male_score is not None and male_score > THRESHOLD:
        return 0, male_score  # Assuming male class is 0
    elif female_score is not None and female_score > THRESHOLD:
        return 1, female_score  # Assuming female class is 1
    else:
        return -1, None  # Class is undetermined or low confidence

def gender_equality_score():
   # Hugging Face pipeline for gender classification
    pipe = pipeline("image-classification", model="rizvandwiki/gender-classification-2")
    
    # Define a threshold for classification confidence
    THRESHOLD = 0.7  
    
    OUTPUT_EXCEL = "classification_results.xlsx"

    def _fn(images, prompts, metadata):
        scores = []
        classifications = []
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        for image in images:
            classification, score = classify_image(image, pipe, THRESHOLD)
            classifications.append(classification)
            # Handle undetermined cases
            scores.append(score if score is not None else 0.5)
            
        classifications = torch.tensor(classifications)
        scores = torch.tensor(scores)

        # classification model returns 0 for male and 1 for female
        female_count = (classifications == 1).sum().item()
        male_count = (classifications == 0).sum().item()

        # Calculate the ratio of women to men
        # count ratio of women to men ASSUMING WOMEN CLASS IS 1
        ratio = female_count / (female_count + male_count) if female_count + male_count > 0 else 0.0
        dist_to_equal = abs(ratio - 0.5)
        #return positive reward to the minority class and negative reward to the majority class
        if ratio < 0.5:
            rewards = torch.where(classifications == 1, dist_to_equal, -dist_to_equal)
        else:
            rewards = torch.where(classifications == 0, dist_to_equal, -dist_to_equal)
        
        return rewards, {}
    
    return _fn

def gender_equality_score_const(underrep='female'):
   # Hugging Face pipeline for gender classification
    pipe = pipeline("image-classification", model="rizvandwiki/gender-classification-2")
    
    def _fn(images, prompts, metadata):
        scores = []
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        for image in images:
            result = pipe(image)
            score = next((item['score'] for item in result if item['label'] == underrep), None)
            scores.append(score)
            
        rewards = torch.tensor(scores)

        return rewards, {}
    
    return _fn


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]] * len(image_batch),
                "answers": [[f"The image contains {prompt}"] for prompt in prompt_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += np.array(response_data["precision"]).squeeze().tolist()
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
