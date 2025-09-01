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


class Draw:
    def __init__(self, source, deck_list=None, debug=False):
        self.decklist = None
        if deck_list is not None:
            with open(deck_list) as f:
                self.decklist = [line.rstrip() for line in f.readlines()]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = hf_hub_download(repo_id="HichTala/draw2", filename="draw_config.json")
        with open(config, "rb") as f:
            self.configs = json.load(f)
        yolo_path = hf_hub_download(repo_id="HichTala/draw2", filename="ygo_yolo.pt")

        model_regression = YOLO(yolo_path)
        self.results = model_regression.track(
            source=source,
            show_labels=False,
            save=False,
            device=device,
            stream=True,
            verbose=False
        )

        image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            use_fast=True
        )
        self.classifier = pipeline(
            "image-classification",
            model="HichTala/draw2",
            image_processor=image_processor,
            device_map=device
        )

        self.dataset = load_dataset("HichTala/ygoprodeck-dataset", split="train")
        labels = self.dataset.features["label"].names
        self.label2id = dict()
        for i, label in enumerate(labels):
            self.label2id[label] = str(i)

        self.debug_mode = debug
        self.cards_prob = {}



    def process(self, result, show=False, display=False):
        outputs = {}

        if display or self.debug_mode:
            outputs['predictions'] = []
        if show or self.debug_mode:
            outputs['image'] = result.orig_img.copy()

        for nbox, boxe in enumerate(result.obb.xyxyxyxyn):
            boxe = np.float32(
                [[b[0] * result.orig_img.shape[1], b[1] * result.orig_img.shape[0]] for b in boxe.cpu()]
            )
            obb = np.intp(boxe)
            xy1, _, xy2, _ = obb

            max_aspect_ratio = self.configs['max_aspect_ratio']
            min_aspect_ratio = self.configs['min_aspect_ratio']
            aspect_ratio = max(
                result.obb.xywhr[nbox][2], result.obb.xywhr[nbox][3]
            ) / min(
                result.obb.xywhr[nbox][2], result.obb.xywhr[nbox][3]
            )

            if self.debug_mode and show:
                cv2.drawContours(outputs['image'], [obb], 0, (119, 152, 255), 2)
                cv2.putText(outputs['image'], "aspect ratio :" + str(aspect_ratio), (xy1[0], xy2[1]),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0,
                            (255, 255, 255),
                            2)

            if max_aspect_ratio > aspect_ratio > min_aspect_ratio:
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
                    contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]
                    box_txt, txt_aspect_ratio = utils.get_txt(contour)

                    max_txt_aspect_ratio = self.configs["max_txt_aspect_ratio"]
                    min_txt_aspect_ratio = self.configs["min_txt_aspect_ratio"]
                    max_txt_area = self.configs["max_txt_area"]
                    min_txt_area = self.configs["min_txt_area"]

                    text_area = cv2.contourArea(box_txt)

                    if self.debug_mode and show:
                        cv2.putText(outputs['image'],
                                    f"text area : {str(text_area)}",
                                    (xy1[0], xy1[1] + (xy2[1] - xy1[1]) // 2),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1.0,
                                    (255, 255, 255),
                                    2)
                        cv2.putText(outputs['image'],
                                    f"text aspect ratio : {str(txt_aspect_ratio)}",
                                    (xy1[0], xy1[1] + (xy2[1] - xy1[1]) // 4),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1.0,
                                    (255, 255, 255),
                                    2)

                    if result.obb.id[nbox].item() not in self.cards_prob.keys() or self.cards_prob[result.obb.id[nbox].item()] >= self.th:
                        rotation = utils.get_rotation(boxes=result.obb.xywhr[nbox], box_txt=box_txt)
                        if rotation is None:
                            break

                        if rotation != 0:
                            roi = cv2.rotate(roi, rotation)

                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi = Image.fromarray(roi)

                        output = self.classifier(roi, top_k=15)
                        if result.obb.id[nbox].item() not in self.cards_prob.keys():
                            self.cards_prob[result.obb.id[nbox].item()] = {card['label']: card['score'] for card in output}
                        else:
                            card_prob = self.cards_prob[result.obb.id[nbox].item()]
                            self.cards_prob[result.obb.id[nbox].item()] = {
                                card['label']: 0.5 * card_prob[card['label']] + 0.5 * card['score'] if card['label'] in card_prob else 0 for card in output
                            }

                        if self.decklist is None:
                            if display:
                                outputs['predictions'].append(output[0]['label'])
                            if self.debug_mode:
                                outputs['predictions'].append(output)
                            if show:
                                cv2.putText(outputs['image'], ' '.join(output[0]['label'].split('-')[:-1]),
                                            (xy1[0], xy1[1]),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            1.0,
                                            (255, 255, 255),
                                            2)
                        else:
                            for label in output:
                                label = label['label']
                                if label.split('-')[-1] in self.decklist:
                                    if display:
                                        outputs['predictions'].append(label)
                                    if show:
                                        cv2.putText(outputs['image'], ' '.join(label.split('-')[:-1]), (xy1[0], xy1[1]),
                                                    cv2.FONT_HERSHEY_PLAIN,
                                                    1.0,
                                                    (255, 255, 255),
                                                    2)
                                    break
                        cv2.drawContours(outputs['image'], [obb], 0, (152, 255, 119), 2)
        return outputs
