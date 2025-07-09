import struct
import time
from collections import deque

import posix_ipc

from draw.draw import Draw
from draw.utils import read_shared_frame, get_deck_list, send_image_to_obs

OBS_SHM_NAME = "/obs_shared_memory"
PYTHON_SHM_NAME = "/python_shared_memory"
HEADER_FORMAT = "II"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class DrawSharedMemoryHandler:
    def __init__(self):
        self.draw = Draw(deck_list=get_deck_list("/home/hicham/Downloads/Odion_FS_Primite.ydk"))

        self.minimum_out_of_screen_time = 25
        self.minimum_screen_time = 6
        self.displayed_time = 0
        self.displayed = {}
        self.counts = {}
        self.queue = deque([])

        self.waiting = True

    def __call__(self):
        while True:
            try:
                shm = posix_ipc.SharedMemory("/obs_shared_memory", flags=0)
            except posix_ipc.ExistentialError:
                continue
            except KeyboardInterrupt:
                break
            image = read_shared_frame(shm, HEADER_SIZE, HEADER_FORMAT)
            if image.size[0] == 0 or image.size[1] == 0:
                continue
            try:
                results = self.draw.model_regression.track(
                    source=image,
                    show_labels=False,
                    save=False,
                    device=self.draw.device,
                    verbose=False,
                    persist=True
                )
                for result in results:
                    outputs = self.draw.process(result, display=True)
                    self.display_card(outputs)
            except Exception as e:
                print(f"Error processing shared memory: {e}")
                breakpoint()

    def display_card(self, outputs):
        for label in outputs['predictions']:
            if label not in self.counts:
                self.counts[label] = 0
            self.counts[label] += 1

        if len(self.queue) and time.time() - self.displayed_time > self.minimum_screen_time:
            label = self.queue.popleft()
            self.displayed[label] = time.time()
            self.displayed_time = time.time()
            image = self.draw.dataset[int(self.draw.label2id[label])]["image"]
            send_image_to_obs(image, PYTHON_SHM_NAME, HEADER_SIZE, HEADER_FORMAT)

        for label, count in self.counts.items():
            if count > 60:
                if label not in self.displayed:
                    if time.time() - self.displayed_time > self.minimum_screen_time:
                        self.displayed[label] = time.time()
                        self.displayed_time = time.time()
                        image = self.draw.dataset[int(self.draw.label2id[label])]["image"]
                        send_image_to_obs(image, PYTHON_SHM_NAME, HEADER_SIZE, HEADER_FORMAT)
                    else:
                        if label not in self.queue:
                            self.queue.append(label)
                else:
                    if time.time() - self.displayed[label] > self.minimum_out_of_screen_time:
                        del self.displayed[label]
                        self.counts[label] = 0


def run():
    sh_memory_handler = DrawSharedMemoryHandler()
    sh_memory_handler()


if __name__ == '__main__':
    run()
