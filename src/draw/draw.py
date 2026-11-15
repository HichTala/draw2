import json
import os

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, pipeline
from ultralytics import YOLO

from draw import utils


def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)


def camera_frames(index, width, height):
    """Read frames from a webcam at the requested resolution.

    Ultralytics opens webcams with OpenCV's default size (often 640x480).
    Opening the camera here lets us ask for a higher resolution; the frames
    are still resized for the detector, but the displayed/saved image and
    the card crops keep the full resolution.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    # Many webcams only reach high resolutions at a usable FPS with MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual != (width, height):
        print(f"Camera {index} does not support {width}x{height}, using {actual[0]}x{actual[1]}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


class Draw:
    def __init__(self, source, deck_lists=None, confidence_threshold=5, camera_resolution=None):
        self.decklist = []
        if deck_lists is not None:
            for deck_list in deck_lists:
                with open(deck_list) as f:
                    self.decklist += [line.rstrip() for line in f.readlines()]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = hf_hub_download(repo_id="HichTala/draw2", filename="draw_config.json")
        yolo_path = hf_hub_download(repo_id="HichTala/draw2", filename="ygo_yolo.pt")
        cardnames_path = hf_hub_download(repo_id="HichTala/draw2", filename="cardnames.json")
        with open(config, "rb") as f:
            self.configs = json.load(f)
        with open(cardnames_path, "r", encoding="utf-8") as f:
            self.cardnames = json.load(f)

        model_regression = YOLO(yolo_path)
        predict_args = dict(
            show_labels=False,
            save=False,
            device=self.device,
            verbose=False
        )
        if camera_resolution is not None and str(source).isdigit():
            frames = camera_frames(int(source), *camera_resolution)
            self.results = (
                result
                for frame in frames
                for result in model_regression.predict(source=frame, **predict_args)
            )
        else:
            self.results = model_regression.predict(source=source, stream=True, **predict_args)

        image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            use_fast=True
        )
        self.classifier = pipeline(
            "image-classification",
            model="HichTala/draw2",
            image_processor=image_processor,
            device_map=self.device
        )

        self.dataset = load_dataset("HichTala/ygoprodeck-dataset", split="train")
        labels = self.dataset.features["label"].names
        self.label2id = dict()
        for i, label in enumerate(labels):
            self.label2id[label] = str(i)

        self.confidence_threshold = confidence_threshold

    def process(self, result, show=False, display=False, language='EN'):
        outputs = {}

        if display:
            outputs['predictions'] = []
        if show:
            outputs['image'] = result.orig_img.copy()

        # if result.obb.id is not None:
        for nbox, boxe in enumerate(result.obb.xyxyxyxyn):
            boxe = np.float32(
                [[b[0] * result.orig_img.shape[1], b[1] * result.orig_img.shape[0]] for b in boxe.cpu()]
            )
            obb = np.intp(boxe)
            xy1, _, xy2, _ = obb

            output_pts = np.float32([
                [224, 224],
                [224, 0],
                [0, 0],
                [0, 224]
            ])
            perspective_transform = cv2.getPerspectiveTransform(boxe, output_pts)
            roi = cv2.warpPerspective(
                result.orig_img, perspective_transform, (224, 224), flags=cv2.INTER_LINEAR
            )
            contours = utils.extract_contours(
                roi,
                d=self.configs["bilateral_filter_d"],
                sigma_color=self.configs["bilateral_filter_sigma_color"],
                sigma_space=self.configs["bilateral_filter_sigma_space"],
                thresh=self.configs["txt_box_contour_threshold"]
            )

            if contours != ():

                if show:
                    cv2.drawContours(outputs['image'], [obb], 0, (152, 255, 119), 2)

                contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]
                box_txt, txt_aspect_ratio = utils.get_txt(contour)

                rotation = utils.get_rotation(boxes=result.obb.xywhr[nbox], box_txt=box_txt)
                if rotation is None:
                    continue

                if rotation != 0:
                    roi = cv2.rotate(roi, rotation)

                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = Image.fromarray(roi)

                output = self.classifier(roi, top_k=15)
                if output[0]['score'] >= self.confidence_threshold / 100:
                    if len(self.decklist) == 0:
                        if display:
                            outputs['predictions'].append(output[0]['label'])
                        if show:
                            card_id = output[0]['label'].split('-')[-1]
                            card_name = self.cardnames.get(card_id, {}).get(language) or ' '.join(output[0]['label'].split('-')[:-1])
                            outputs['image'] = utils.put_text(outputs['image'], card_name,
                                        (xy1[0], xy1[1]))

                    else:
                        for label in output:
                            label = label['label']
                            if label.split('-')[-1] in self.decklist:
                                if display:
                                    outputs['predictions'].append(output[0]['label'])
                                if show:
                                    card_id = output[0]['label'].split('-')[-1]
                                    card_name = self.cardnames.get(card_id, {}).get(language) or ' '.join(output[0]['label'].split('-')[:-1])
                                    outputs['image'] = utils.put_text(outputs['image'], card_name,
                                                (xy1[0], xy1[1]))
                                break
        return outputs
