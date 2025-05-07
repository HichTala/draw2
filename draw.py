import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

import tools
from build_models import build_regression, build_classification


class Draw:
    def __init__(self, config, deck_list, source, debug=False):
        with open(config, "rb") as f:
            self.configs = json.load(f)
        with open(deck_list) as f:
            self.deck_list = [line.rstrip() for line in f.readlines()]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.card_types = self.configs['card_types']
        model_regression = build_regression(os.path.join(self.configs['trained_models'], 'yolo_ygo.pt'))
        self.classifier, self.deck_card_ids = build_classification(
            card_types=self.card_types,
            configs=self.configs,
            data_path=self.configs['data_path'],
            deck_list=self.deck_list,
            device=device
        )

        if source == '0':
            source = str(self.configs['webcam'])

        self.results = model_regression(
            source=source,
            show_labels=False,
            save=False,
            device=device,
            stream=True,
            verbose=False
        )

        self.debug_mode = debug

    def process(self, result, display=False):
        predictions = []

        image = result.orig_img.copy()
        for nbox, boxe in enumerate(result.obb.xyxyxyxyn):
            max_aspect_ratio = 1.90
            min_aspect_ratio = 1.00
            aspect_ratio = max(
                result.obb.xywhr[nbox][2], result.obb.xywhr[nbox][3]
            ) / min(
                result.obb.xywhr[nbox][2], result.obb.xywhr[nbox][3]
            )

            # if self.debug_mode:
            #     cv2.putText(image, "box area :" + str(np.abs((x1 - x2) * (y1 - y2))), (x1, y2),
            #                 cv2.FONT_HERSHEY_PLAIN,
            #                 1.0,
            #                 (255, 255, 255),
            #                 2)

            if max_aspect_ratio > aspect_ratio > min_aspect_ratio:
                boxe = np.float32([[b[0] * 1920, b[1] * 1080] for b in boxe.cpu()])
                output_pts = np.float32([
                    [224, 224],
                    [224, 0],
                    [0, 0],
                    [0, 224]
                ])
                perspective_transform = cv2.getPerspectiveTransform(boxe, output_pts)
                roi = cv2.warpPerspective(image, perspective_transform, (224, 224), flags=cv2.INTER_LINEAR)
                contours = tools.extract_contours(roi)

                if contours != ():
                    contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]
                    box_txt, txt_aspect_ratio = tools.get_txt(contour)

                    max_txt_aspect_ratio = 5
                    min_txt_aspect_ratio = 1
                    max_txt_area = 11000
                    min_txt_area = 6000

                    # if self.debug_mode:
                    #     cv2.putText(image, "text area :" + str(area), (x1, y1 + (y2 - y1) // 2),
                    #                 cv2.FONT_HERSHEY_PLAIN,
                    #                 1.0,
                    #                 (255, 255, 255),
                    #                 2)

                    if (
                            (min_txt_aspect_ratio < txt_aspect_ratio < max_txt_aspect_ratio)
                            and
                            (min_txt_area < cv2.contourArea(box_txt) < max_txt_area)
                    ):
                        rotation = tools.get_rotation(boxes=result.obb.xywhr[nbox], box_txt=box_txt)
                        if rotation is None:
                            break

                        if rotation != 0:
                            roi = cv2.rotate(roi, rotation)

                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi = Image.fromarray(roi)

                        # if self.debug_mode:
                        #     cv2.putText(image, "area threshold :" + str(cv2.contourArea(box_artwork)),
                        #                 (x1, y1 + (y2 - y1) // 4),
                        #                 cv2.FONT_HERSHEY_PLAIN,
                        #                 1.0,
                        #                 (255, 255, 255),
                        #                 2)

                        output = self.classifier(roi)

                        # _, indices = torch.sort(output, descending=True)
                        # for k, i in enumerate(indices[0]):
                        #     if i in self.deck_card_ids[card_type]:
                        #         predictions.append((self.classes_dict[card_type][i], card_type))
                        #         cv2.putText(image, self.classes_dict[card_type][i], (x1, y1),
                        #                     cv2.FONT_HERSHEY_PLAIN,
                        #                     1.0,
                        #                     (255, 255, 255),
                        #                     2)
                        #         break
                        # cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 152, 119), thickness=2)
                        # cv2.drawContours(roi, [box_txt], 0, (152, 255, 119), 2)
                        # cv2.drawContours(roi, [box_artwork], 0, (119, 152, 255), 2)
        if display:
            return image
        else:
            return predictions
