import math

from PIL import Image
from transformers import pipeline, AutoImageProcessor

from ultralytics import YOLO
import numpy as np
import cv2


def extract_contours(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    equalized = cv2.equalizeHist(gray)
    # cv2.imshow('image', equalized)
    # cv2.waitKey(0)
    _, thresh = cv2.threshold(equalized, 140, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)

    kernel = np.ones((7, 7), np.uint8)
    edged = cv2.erode(thresh, kernel, iterations=3)
    # cv2.imshow('image', edged)
    # cv2.waitKey(0)
    edged = cv2.dilate(edged, kernel, iterations=3)
    # cv2.imshow('image', edged)
    # cv2.waitKey(0)

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours


# Load a model
# model = YOLO("yolov8n.pt")
model = YOLO('runs/obb/train4/weights/best.pt')

# Use the model
# source = "/home/hicham/Downloads/test.mp4"
source = "/home/hicham/Downloads/2023-11-09 12-11-48.mp4"
results = model.track(source=source,
                      save=False,
                      show=False,
                      # show_labels=True,
                      # show_conf=True,
                      # conf=0.85,
                      verbose=True,
                      stream=True,
                      # line_width=2,
                      imgsz=640)

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
classifier = pipeline("image-classification", model="HichTala/draw", image_processor=image_processor)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter('filename.avi', fourcc, 120, (1920, 1080))

do_break = False

try:
    for nbox, boxes in enumerate(results):
        # if nbox <= 4352:
        #     continue
        orig_image = boxes.orig_img
        try:
            for n, boxe in enumerate(boxes.obb.xyxyxyxyn):
                max_aspect_ratio = 1.90
                min_aspect_ratio = 1.00
                aspect_ratio = max(
                    boxes.obb.xywhr[n][2], boxes.obb.xywhr[n][3]
                ) / min(
                    boxes.obb.xywhr[n][2], boxes.obb.xywhr[n][3]
                )
                if do_break:
                    obb = np.float32([[b[0] * 1920, b[1] * 1080] for b in boxe.cpu()])
                    obb = np.intp(obb)
                    cv2.drawContours(orig_image, [obb], 0, (119, 152, 255), 2)
                    cv2.imwrite("orig_image.png", orig_image)
                    print("aspect ratio ", aspect_ratio)
                    breakpoint()
                if max_aspect_ratio > aspect_ratio > min_aspect_ratio:
                    # boxe = boxes.obb.xyxyxyxyn[0]
                    boxe = np.float32([[b[0] * 1920, b[1] * 1080] for b in boxe.cpu()])
                    # rect = cv2.boundingRect(boxe)
                    # x, y, w, h = rect
                    # cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # cv2.drawContours(orig_image, [boxe], 0, (119, 152, 255), 2)

                    # height_1 = np.sqrt(((boxe[0][1] - boxe[1][1]) ** 2) + ((boxe[0][0] - boxe[1][0]) ** 2))
                    # height_2 = np.sqrt(((boxe[2][1] - boxe[3][1]) ** 2) + ((boxe[2][0] - boxe[3][0]) ** 2))
                    # width_1 = np.sqrt(((boxe[0][1] - boxe[2][1]) ** 2) + ((boxe[0][0] - boxe[2][0]) ** 2))
                    # width_2 = np.sqrt(((boxe[1][1] - boxe[3][1]) ** 2) + ((boxe[1][0] - boxe[3][0]) ** 2))
                    #
                    # max_height = max(int(height_1), int(height_2))
                    # max_width = max(int(width_1), int(width_2))

                    output_pts = np.float32([
                        [224, 224],
                        [224, 0],
                        [0, 0],
                        [0, 224]
                    ])
                    M = cv2.getPerspectiveTransform(boxe, output_pts)
                    out = cv2.warpPerspective(orig_image, M, (224, 224), flags=cv2.INTER_LINEAR)

                    contours = extract_contours(out.copy())
                    if contours != ():
                        contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]

                        rect = cv2.minAreaRect(contour)
                        box_txt = cv2.boxPoints(rect)
                        box_txt = np.intp(box_txt)

                        dx = max(box_txt[:, 0]) - min(box_txt[:, 0])
                        dy = max(box_txt[:, 1]) - min(box_txt[:, 1])
                        txt_aspect_ratio = max(dx, dy) / min(dx, dy)

                        if do_break:
                            print("ratio ", txt_aspect_ratio)
                            print("area ", cv2.contourArea(box_txt))
                            cv2.imwrite(f"images/image_visu.png", out)
                            breakpoint()
                        if 2.5 < txt_aspect_ratio < 5 and 6000 < cv2.contourArea(box_txt) < 11000:
                            w = boxes.obb.xywhr[n][2]
                            h = boxes.obb.xywhr[n][3]
                            angle = boxes.obb.xywhr[n][4] % math.pi / 2
                            if min(box_txt[:, 0]) < 112:  # on élimine 90 clockwise
                                if max(box_txt[:, 0]) < 112:  # on vérifie 90 anticlockwise
                                    if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
                                        if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                                            r = 270
                                        else:
                                            r = None
                                    else:
                                        r = None
                                else:  # on élimine les 90
                                    if min(box_txt[:, 1]) < max(box_txt[:, 1]) < 112:
                                        if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                                            r = 180
                                        else:
                                            r = None
                                    elif 112 < min(box_txt[:, 1]) < max(box_txt[:, 1]):
                                        if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                                            r = 0
                                        else:
                                            r = None
                                    else:
                                        r = None
                            else:  # on vérifie 90 clockwise
                                if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
                                    if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                                        r = 90
                                    else:
                                        r = None
                                else:
                                    r = None

                            # print("rotation ", r)
                            if r is not None:
                                out = cv2.drawContours(out, [box_txt], 0, (152, 255, 119), 2)
                                rotation = {
                                    90: cv2.ROTATE_90_CLOCKWISE,
                                    180: cv2.ROTATE_180,
                                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                                }
                                if r != 0:
                                    out = cv2.rotate(out, rotation[r])
                                cv2.imwrite(f"images/image.png", out)

                                image = Image.open(f"images/image.png")
                                results = classifier(image)
                                cv2.putText(orig_image, results[0]['label'], (int(boxe[0, 0]), int(boxe[0, 1])),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            1.0,
                                            (255, 255, 255),
                                            2)

                                boxe = np.intp(boxe)
                                orig_image = cv2.drawContours(orig_image, [boxe], 0, (255, 152, 119), 2)
                                # breakpoint()
        except KeyboardInterrupt:
            do_break = True
            breakpoint()

        cv2.imwrite("orig_image.png", orig_image)
        # cv2.imshow("orig_image.png", orig_image)
        # cv2.waitKey(1)
        # breakpoint()

        video_writer.write(orig_image)
    video_writer.release()

except KeyboardInterrupt:
    video_writer.release()
