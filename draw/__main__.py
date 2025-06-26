import asyncio
import base64
import time
from collections import deque

import cv2
import numpy as np
from websockets import serve

from draw.draw import Draw


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

    async def handler(self, websockets):
        async for message in websockets:
            if message["type"] == "image" and self.waiting:
                # TODO: get the image in the right format
                image = base64.b64decode(message['image'])
                results = self.draw.model_regression.track(
                    source=image,
                    show_labels=False,
                    save=False,
                    device=self.draw.device,
                    verbose=False,
                    persist=True
                )

                outputs = self.draw.process(results, display=True)
                self.display_card(websockets, outputs)

    async def main(self):
        async with serve(self.handler, "localhost", 1996) as server:
            await server.serve_forever()

    def display_card(self, websockets, outputs):
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
            websockets.send(image)

        for label, count in self.counts.items():
            if count > 60:
                if label not in self.displayed:
                    if time.time() - self.displayed_time > self.minimum_screen_time:
                        self.displayed[label] = time.time()
                        self.displayed_time = time.time()
                        image = self.draw.dataset[int(self.draw.label2id[label])]["image"]
                        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        # TODO: send image in the right format
                        websockets.send(image)
                    else:
                        self.queue.append(label)
                else:
                    if time.time() - self.displayed[label] > self.minimum_out_of_screen_time:
                        del self.displayed[label]
                        self.counts[label] = 0


if __name__ == '__main__':
    websockets_handler = DrawWebSocketHandler()
    asyncio.run(websockets_handler.main())
