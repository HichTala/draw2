import os

import numpy as np
import torch
from scipy import interpolate
from timm import create_model
from transformers import AutoImageProcessor, pipeline
from ultralytics import YOLO

from src.tools import clean_deck_list


def build_regression(yolo_path):
    return YOLO(yolo_path)


def build_classification(card_types, configs, data_path, deck_list, device):
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    classifier = pipeline("image-classification", model="HichTala/draw", image_processor=image_processor)

    deck_card_ids = clean_deck_list(deck_list)

    return classifier, deck_card_ids

