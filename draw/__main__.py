import asyncio
import base64
import socket
import struct
import threading
import time
from collections import deque

import cv2
import numpy as np
from websockets import serve

from draw.draw import Draw

HOST = 'localhost'
PORT = 1996


class DrawWebSocketHandler:
    def __init__(self):
        self.draw = Draw()

        self.minimum_out_of_screen_time = 5
        self.minimum_screen_time = 70
        self.displayed_time = 0
        self.displayed = {}
        self.counts = {}
        self.queue = deque([])

        self.waiting = True

    def __call__(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print(f"[INFO] Server listening on {HOST}:{PORT}")

            while True:
                conn, addr = s.accept()
                client_thread = threading.Thread(target=self.handler, args=(conn, addr), daemon=True)
                client_thread.start()

    def handler(self, conn, addr):
        print(f"[INFO] Connected by {addr}")

        try:
            while True:
                size_data = conn.recv(4)
                if not size_data:
                    break

                img_size = struct.unpack('!I', size_data)[0]
                img_data = b''

                while len(img_data) < img_size:
                    packet = conn.recv(img_size - len(img_data))
                    if not packet:
                        break
                    img_data += packet

                print(f"[DEBUG] Received image of size {len(img_data)} bytes")
                if self.waiting:
                    # TODO: get the image in the right format
                    image = base64.b64decode(img_data)
                    results = self.draw.model_regression.track(
                        source=image,
                        show_labels=False,
                        save=False,
                        device=self.draw.device,
                        verbose=False,
                        persist=True
                    )

                    outputs = self.draw.process(results, display=True)
                    self.display_card(conn, outputs)

        except Exception as e:
            print(f"[ERROR] {e}")

        finally:
            print(f"[INFO] Connection closed {addr}")
            conn.close()

    def display_card(self, conn, outputs):
        for label in outputs['predictions']:
            if label not in self.counts:
                self.counts[label] = 0
            self.counts[label] += 1

        label = self.queue.popleft()
        if label is not None:
            self.displayed[label] = time.time()
            image = self.draw.dataset[int(self.draw.label2id[label])]["image"]
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # TODO: send image in the right format
            conn.sendall(image)

        for label, count in self.counts.items():
            if count > 60:
                if label not in self.displayed:
                    if time.time() - self.displayed_time > self.minimum_screen_time:
                        self.displayed[label] = time.time()
                        self.displayed_time = time.time()
                        image = self.draw.dataset[int(self.draw.label2id[label])]["image"]
                        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        # TODO: send image in the right format
                        conn.sendall(image)
                    else:
                        self.queue.append(label)
                else:
                    if time.time() - self.displayed[label] > self.minimum_out_of_screen_time:
                        del self.displayed[label]
                        self.counts[label] = 0


if __name__ == '__main__':
    websockets_handler = DrawWebSocketHandler()
    websockets_handler()
