import argparse
import os
import platform
import shutil
import time
import urllib
from pathlib import Path

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from draw.draw import Draw


def parse_command_line():
    parser = argparse.ArgumentParser('Detecting and Recognizing a Wide Range of Yu-Gi-Oh! Cards', add_help=True)

    parser.add_argument('--source', default='0', type=str,
                        help="Source to use for processing refer to github.com/HichTala/draw2 for more details")

    parser.add_argument("--save", nargs="?", const="output",
                        help="Save the video. Optionally provide a path. If not, saves as 'output' in current dir.")

    parser.add_argument('--show', action='store_true',
                        help="Show video.")
    parser.add_argument('--display-card', action='store_true',
                        help="Display one of the cards that appear at a time.")

    parser.add_argument('--deck-list', default=None, type=str,
                        help="Path or link to the deck list.")

    parser.add_argument('--fps', default=60, type=int,
                        help="FPS of the saved video.")

    return parser.parse_args()


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


def display_card(outputs, counts, displayed, dataset, label2id):
    for label in outputs['predictions']:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    for label, count in counts.items():
        if count > 60:
            if label not in displayed:
                displayed[label] = 6
                image = dataset[int(label2id[label])]["image"]
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                show(image, p="draw2 - Card")
            else:
                displayed[label] -= 1
                if displayed[label] == 0:
                    del displayed[label]
                    counts[label] = 0


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


def get_deck_list(deck_list):
    if os.path.isfile(deck_list):
        return deck_list
    elif deck_list.isdigit() and len(deck_list) == 8:
        options = Options()
        options.add_argument('--headless')
        options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

        url = f"https://www.duelingbook.com/deck?id={deck_list}"

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(3)
        driver.get(url)
        time.sleep(1)

        for entry in driver.get_log('browser'):
            deck_name = parse_deck_name(console_entry=entry) or 'No name'
            if deck_name == 'No name':
                continue

            cache_dir = get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            local_path = cache_dir / f"{deck_name}.ydk"

            db_list = parse_deck_list(entry.get("message"), [])
            db_list = list(set(db_list))

            with open(local_path, 'w') as f:
                for line in db_list:
                    f.write(f"{line}\n")
            return local_path

        return None
    else:
        cache_dir = get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / f"{deck_list.split("/")[-1]}.ydk"

        if not local_path.exists():
            print(f"Downloading {deck_list} to {local_path}")
            urllib.request.urlretrieve(deck_list, local_path)
        else:
            print(f"Using cached: {local_path}")

        return local_path


def main(args):
    deck_list = None
    if args.deck_list:
        deck_list = get_deck_list(args.deck_list)

    draw = Draw(
        source=args.source,
        deck_list=deck_list,
        # debug=True
    )

    is_image = False
    image_types = ["bmp", "dng", "HEIC", "jpeg", "jpg", "mpo", "pfm", "png", "tif", "tiff", "webp"]
    if args.save and args.source is not None and args.source.split('.')[-1] in image_types:
        is_image = True
        save_path = args.save if args.save.split(".")[-1] in image_types else "output.png"

    if args.save and not is_image:
        video_types = ["asf", "avi", "gif", "m4v", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"]
        first_frame = next(iter(draw.results)).orig_img
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        save_path = ".".join(args.save.split('.')[:-1]) if args.save.split(".")[-1] in video_types else "output"

        video_writer = cv2.VideoWriter(f'{save_path}.avi', fourcc, args.fps,
                                       (first_frame.shape[0], first_frame.shape[1]))

    if args.display_card:
        # TODO put those as arguments inside draw class
        counts = {}
        displayed = {}

        labels = draw.dataset.features["label"].names
        label2id = dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)

    try:
        for result in draw.results:
            outputs = draw.process(result, show=args.show, display=args.display_card)

            if args.show:
                show(outputs['image'])
                # result.show()

            if args.save:
                save(is_image, outputs, video_writer, save_path)

            if args.display_card:
                display_card(outputs, counts, displayed, draw.dataset, label2id)

            if draw.debug_mode and outputs['predictions'] != []:
                print(outputs['predictions'])

        cv2.destroyAllWindows()
        if args.save and not is_image:
            video_writer.release()
        clear_cache()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        if args.save and not is_image:
            video_writer.release()
        clear_cache()


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
