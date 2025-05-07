import json
import os

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, pipeline
from ultralytics import YOLO

from draw import utils


def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)


class Draw:
    def __init__(self, config, source, deck_list=None, debug=False):
        with open(config, "rb") as f:
            self.configs = json.load(f)
        with open(deck_list) as f:
            self.deck_list = [line.rstrip() for line in f.readlines()]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #TODO: Use huggingface hub to download the model
        if not os.path.isfile(self.configs["yolo_path"]):
            download_file(self.configs["hf_yolo"], self.configs["yolo_path"])

        model_regression = YOLO(self.configs["yolo_path"])
        self.results = model_regression.track(
            source=source,
            show_labels=False,
            save=False,
            device=device,
            stream=True,
            verbose=False
        )

        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = pipeline(
            "image-classification",
            model="HichTala/draw",
            image_processor=image_processor,
            device_map=device
        )

        self.debug_mode = debug

    def process(self, result, show=False, display=False):
        outputs = {}

        if show:
            outputs['predictions'] = []
        if display:
            outputs['image'] = result.orig_img.copy()

        for nbox, boxe in enumerate(result.obb.xyxyxyxyn):
            x1, y1, _, _, x2, y2, _, _ = boxe.cpu().numpy().astype(int)

            max_aspect_ratio = self.configs['max_aspect_ratio']
            min_aspect_ratio = self.configs['min_aspect_ratio']
            aspect_ratio = max(
                result.obb.xywhr[nbox][2], result.obb.xywhr[nbox][3]
            ) / min(
                result.obb.xywhr[nbox][2], result.obb.xywhr[nbox][3]
            )

            if self.debug_mode and show:
                cv2.putText(outputs['image'], "aspect ratio :" + str(aspect_ratio), (x1, y2),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0,
                            (255, 255, 255),
                            2)

            if max_aspect_ratio > aspect_ratio > min_aspect_ratio:
                boxe = np.float32(
                    [[b[0] * outputs['image'].shape[1], b[1] * outputs['image'].shape[0]] for b in boxe.cpu()]
                )
                output_pts = np.float32([
                    [224, 224],
                    [224, 0],
                    [0, 0],
                    [0, 224]
                ])
                perspective_transform = cv2.getPerspectiveTransform(boxe, output_pts)
                roi = cv2.warpPerspective(
                    outputs['image'], perspective_transform, (224, 224), flags=cv2.INTER_LINEAR
                )
                contours = utils.extract_contours(roi)

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
                                    (x1, y1 + (y2 - y1) // 2),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1.0,
                                    (255, 255, 255),
                                    2)
                        cv2.putText(outputs['image'],
                                    f"text aspect ratio : {str(txt_aspect_ratio)}",
                                    (x1, y1 + (y2 - y1) // 4),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1.0,
                                    (255, 255, 255),
                                    2)

                    if (
                            (min_txt_aspect_ratio < txt_aspect_ratio < max_txt_aspect_ratio)
                            and
                            (min_txt_area < text_area < max_txt_area)
                    ):
                        rotation = utils.get_rotation(boxes=result.obb.xywhr[nbox], box_txt=box_txt)
                        if rotation is None:
                            break

                        if rotation != 0:
                            roi = cv2.rotate(roi, rotation)

                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi = Image.fromarray(roi)

                        output = self.classifier(roi)

                        for label in output[:]['label']:
                            if label.split('-')[-1] in self.deck_list:
                                outputs['predictions'].append(label)
                                cv2.putText(outputs['image'], ' '.join(label.split('-')[:-2]), (x1, y1),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            1.0,
                                            (255, 255, 255),
                                            2)
                                break
                        cv2.rectangle(outputs['image'], (x1, y1), (x2, y2), color=(255, 152, 119), thickness=2)
        return outputs
