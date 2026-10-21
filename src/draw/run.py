import struct
import time
from collections import deque
from multiprocessing import shared_memory
import mmap
import numpy as np

from draw.draw import Draw
from draw.utils import read_shared_frame, get_deck_list

import sys

OBS_SHM_NAME = "/obs_shared_memory"
PYTHON_SHM_NAME = "/python_shared_memory"


def posix_shm_name(name):
    # SharedMemory already prepends "/" on POSIX, so we'd end up with
    # "//obs_shared_memory_1". Linux doesn't care, macOS does.
    return name.lstrip("/")


HEADER_FORMAT = "II"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

MAX_FRAME_WIDTH = 3840
MAX_FRAME_HEIGHT = 2160
BYTES_PER_PIXEL = 4
FRAME_BUFFER_SIZE = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * BYTES_PER_PIXEL

SIZE = HEADER_SIZE + FRAME_BUFFER_SIZE


class DrawSharedMemoryHandler:
    def __init__(
            self,
            model_id="HichTala/draw2",
            deck_list="",
            minimum_out_of_screen_time=25,
            minimum_screen_time=6,
            confidence_threshold=5,
            channel=1
    ):
        self.draw = Draw(model=model_id, deck_lists=get_deck_list(deck_list), confidence_threshold=confidence_threshold
    )

        self.obs_shm = None
        self.python_shm = None
        self.shm_array = None
        self.channel = channel

        self.minimum_out_of_screen_time = minimum_out_of_screen_time
        self.minimum_screen_time = minimum_screen_time
        self.displayed_time = 0
        self.displayed = {}
        self.counts = {}
        self.queue = deque([])

        self.confidence_threshold = confidence_threshold

        self.waiting = True

    def __call__(self):
        print(f"Waiting for OBS to start...")
        while True:
            try:
                if sys.platform == 'win32':
                    self.obs_shm = mmap.mmap(-1, SIZE, tagname=f"{OBS_SHM_NAME}_{self.channel}", access=mmap.ACCESS_READ)
                    print("mapped size:", self.obs_shm.size())
                else:
                    self.obs_shm = shared_memory.SharedMemory(name=posix_shm_name(f"{OBS_SHM_NAME}_{self.channel}"))
                    print("mapped size:", self.obs_shm.size)
            except (FileNotFoundError, ValueError, OSError):
                continue
            except KeyboardInterrupt:
                return
            break

        print(f"Shared memory found")
        if sys.platform == 'win32':
            obs_buf = memoryview(self.obs_shm)
        else:
            obs_buf = memoryview(self.obs_shm.buf)

        while True:
            try:
                image = read_shared_frame(obs_buf, HEADER_SIZE, HEADER_FORMAT)
            except TypeError as e:
                print("type error: ", e)
                break
            except ValueError as e:
                print("value error: ", e)
                break
            except KeyboardInterrupt:
                print("Stopping Draw2...")
                close_shared_memory(self.obs_shm)
                close_shared_memory(self.python_shm)
                return

            if image.size[0] == 0 or image.size[1] == 0:
                continue

            try:
                results = self.draw.model_regression.predict(
                    source=image,
                    show_labels=False,
                    save=False,
                    device=self.draw.device,
                    verbose=False
                )
                for result in results:
                    outputs = self.draw.process(result)
                    self.display_card(outputs)

            except KeyboardInterrupt:
                print("Stopping Draw2...")
                close_shared_memory(self.obs_shm)
                close_shared_memory(self.python_shm)
                return

    def display_card(self, outputs):
        for label in outputs['predictions']:
            if label in self.displayed:
                if time.time() - self.displayed[label] > self.minimum_out_of_screen_time:
                    del self.displayed[label]
                    if label not in self.queue:
                        self.queue.append(label)
            elif label not in self.queue:
                self.queue.append(label)

        if len(self.queue) and time.time() - self.displayed_time > self.minimum_screen_time:
            label = self.queue.popleft()
            self.displayed[label] = time.time()
            self.displayed_time = time.time()
            image = self.draw.dataset[int(self.draw.label2id[label])]["image"]
            self.send_image_to_obs(image)

    def send_image_to_obs(self, image):
        img_array = np.array(image.convert("RGBA"))
        height, width, channels = img_array.shape

        total_size = HEADER_SIZE + img_array.nbytes

        if sys.platform == 'win32' and (self.python_shm is None or total_size != self.python_shm.size()):
            if self.python_shm is not None:
                self.python_shm.close()
            self.python_shm = mmap.mmap(-1, total_size, tagname=f"{PYTHON_SHM_NAME}_{self.channel}", access=mmap.ACCESS_WRITE)
            self.python_shm.seek(0)
            self.python_shm.write(b"\x00" * total_size)
            python_buf = memoryview(self.python_shm)
            self.shm_array = np.ndarray((height, width, channels), dtype=np.uint8, buffer=python_buf[HEADER_SIZE:])
            python_buf[:HEADER_SIZE] = struct.pack(HEADER_FORMAT, width, height)
        elif sys.platform != 'win32' and (self.python_shm is None or total_size != self.python_shm.size):
            if self.python_shm is not None:
                self.python_shm.close()
                self.python_shm.unlink()
            self.python_shm = shared_memory.SharedMemory(name=posix_shm_name(f"{PYTHON_SHM_NAME}_{self.channel}"), create=True, size=total_size)
            self.shm_array = np.ndarray((height, width, channels), dtype=np.uint8,
                                        buffer=self.python_shm.buf[HEADER_SIZE:])
            self.python_shm.buf[:HEADER_SIZE] = struct.pack(HEADER_FORMAT, width, height)
        self.shm_array[:] = img_array

        print(f"Sent image {width}x{height} to OBS")


def close_shared_memory(shm):
    if sys.platform == 'win32':
        if shm is not None:
            shm.close()
    else:
        if shm is not None:
            shm.close()
            shm.unlink()


def run(
        model_size=0,
        deck_list="",
        minimum_out_of_screen_time=25,
        minimum_screen_time=6,
        confidence_threshold=5,
        channel=1
):
    print("Starting Draw2...")

    sh_memory_handler = DrawSharedMemoryHandler(
        model_id=["HichTala/draw2", "HichTala/draw2-large"][model_size],
        deck_list=deck_list,
        minimum_out_of_screen_time=minimum_out_of_screen_time,
        minimum_screen_time=minimum_screen_time,
        confidence_threshold=confidence_threshold,
        channel=channel
    )
    sh_memory_handler()

    close_shared_memory(sh_memory_handler.obs_shm)
    close_shared_memory(sh_memory_handler.python_shm)


if __name__ == '__main__':
    run()
