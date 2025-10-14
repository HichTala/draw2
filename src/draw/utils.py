import math
import os
import platform
import shutil
import struct
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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


def extract_contours(roi, d, sigma_color, sigma_space, thresh):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    equalized = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(equalized, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    edged = cv2.erode(thresh, kernel, iterations=3)
    edged = cv2.dilate(edged, kernel, iterations=3)

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours


def get_txt(contour):
    rect = cv2.minAreaRect(contour)
    box_txt = cv2.boxPoints(rect)
    box_txt = np.intp(box_txt)

    dx = max(box_txt[:, 0]) - min(box_txt[:, 0])
    dy = max(box_txt[:, 1]) - min(box_txt[:, 1])
    txt_aspect_ratio = max(dx, dy) / min(dx, dy)

    return box_txt, txt_aspect_ratio


def get_rotation(boxes, box_txt):
    w = boxes[2]
    h = boxes[3]
    angle = boxes[4] % math.pi / 2
    if min(box_txt[:, 0]) < 112:  # on élimine 90 clockwise
        if max(box_txt[:, 0]) < 112:  # on vérifie 90 anticlockwise
            if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
                if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                    rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
                else:
                    rotation = None
            else:
                rotation = None
        else:  # on élimine les 90
            if min(box_txt[:, 1]) < max(box_txt[:, 1]) < 112:
                if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                    rotation = cv2.ROTATE_180
                else:
                    rotation = None
            elif 112 < min(box_txt[:, 1]) < max(box_txt[:, 1]):
                if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                    rotation = 0
                else:
                    rotation = None
            else:
                rotation = None
    else:  # on vérifie 90 clockwise
        if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
            if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                rotation = cv2.ROTATE_90_CLOCKWISE
            else:
                rotation = None
        else:
            rotation = None

    return rotation


def show(im, p="draw2"):
    """Display an image in a window."""
    if platform.system() == "Linux":
        cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)
    cv2.imshow(p, im)
    cv2.waitKey(1)  # 1 millisecond


def save(is_image, outputs, video_writer, save_path):
    if is_image:
        cv2.imwrite(f"{save_path}.png", outputs['image'])
    else:
        video_writer.write(outputs['image'])


def get_cache_dir():
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "draw2"


def clear_cache():
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def parse_deck_name(console_entry):
    message = console_entry.get('message', '')
    if not message or '\\"action\\":\\"Success\\"' not in message:
        return ''
    pattern = '\\"name\\":\\"'
    start = message.find(pattern) + len(pattern)
    message = message[start:]
    pattern = '\\",'
    end = message.find(pattern)
    message = message[:end]
    return message


def parse_deck_list(message, dl):
    pattern = '\\"serial_number\\":\\"'
    if pattern not in message:
        return dl
    start = message.find(pattern) + len(pattern)
    message = message[start:]
    pattern = '\\",'
    end = message.find(pattern)
    serial_number = message[:end]
    dl.append(serial_number)

    message = message[end:]
    return parse_deck_list(message, dl)


def get_deck_list(deck_lists):
    returned_deck_lists = []
    for deck_list in deck_lists.split(";"):
        if os.path.isfile(deck_list):
            returned_deck_lists.append(deck_list)
    return returned_deck_lists


def read_shared_frame(buf, header_size, header_format):
    header_bytes = buf[:header_size]
    width, height = struct.unpack(header_format, header_bytes)

    img_size = width * height * 4
    if header_size + img_size > len(buf):
        raise RuntimeError(f"Shared memory too small for {width}x{height} image.")

    frame_data = buf[header_size:header_size + img_size]
    frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))

    img = Image.fromarray(frame_array, mode="RGBA").convert("RGB")
    return img



