import cv2
import numpy as np


def clean_deck_list(deck_list, classes):
    deck_card_id = []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for card_id in deck_list:
        if card_id[0] in '0123456789':
            for card_name in class_to_idx.keys():
                if card_id[0] == '0':
                    if card_id[1:] in card_name:
                        deck_card_id.append(class_to_idx[card_name])
                if card_id in card_name:
                    deck_card_id.append(class_to_idx[card_name])
    return list(set(deck_card_id))

def extract_contours(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    equalized = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(equalized, 140, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    edged = cv2.erode(thresh, kernel, iterations=3)
    edged = cv2.dilate(edged, kernel, iterations=3)

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours