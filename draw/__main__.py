import argparse
import os
import platform

import cv2
import mss
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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


def display_card(outputs, counts, displayed, dataset):
    for label in outputs['predictions']:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    for label, count in counts.items():
        if count > 60:
            if label not in displayed:
                displayed[label] = 6

                image = dataset[label]["image"]
                image.show(title="draw2 - Card")
            else:
                displayed[label] -= 1
                if displayed[label] == 0:
                    del displayed[label]
                    counts[label] = 0


def main(args):
    draw = Draw(
        source=args.source,
        deck_list=args.deck_list,
        debug=True
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

    try:
        for result in draw.results:
            outputs = draw.process(result, show=args.show)

            if args.show:
                show(outputs['image'])
                # result.show()

            if args.save:
                save(is_image, outputs, video_writer, save_path)

            if args.display_card:
                display_card(outputs, counts, displayed, draw.dataset)

        if args.save and not is_image:
            video_writer.release()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        if args.save and not is_image:
            video_writer.release()


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
