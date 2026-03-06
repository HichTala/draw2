import ctypes
import struct
import time
from collections import deque
from multiprocessing import shared_memory
import mmap

import numpy as np
from draw.draw import Draw
from draw.utils import read_shared_frame, get_deck_list

import os
import sys
import logging

OBS_SHM_NAME = "obs_shared_memory"
PYTHON_SHM_NAME = "python_shared_memory"
HEADER_FORMAT = "II"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

MAX_FRAME_WIDTH = 3840
MAX_FRAME_HEIGHT = 2160
BYTES_PER_PIXEL = 4  # RGBA8
FRAME_BUFFER_SIZE = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * BYTES_PER_PIXEL

SIZE = HEADER_SIZE + FRAME_BUFFER_SIZE


class DrawSharedMemoryHandler:
    def __init__(self, deck_list="", minimum_out_of_screen_time=25, minimum_screen_time=6, confidence_threshold=5):
        self.draw = Draw(deck_lists=get_deck_list(deck_list), confidence_threshold=confidence_threshold)

        self.obs_shm = None
        self.python_shm = None
        self.shm_array = None

        self.minimum_out_of_screen_time = minimum_out_of_screen_time
        self.minimum_screen_time = minimum_screen_time
        self.displayed_time = 0
        self.displayed = {}
        self.counts = {}
        self.queue = deque([])

        self.confidence_threshold = confidence_threshold

        self.waiting = True

    def __call__(self, address=None, ready_ptr=None):
        continue_execution = True if address is None else address.contents.value
        if ready_ptr is not None:
            ready_ptr.contents.value = True

        print(f"Waiting for OBS to start...")
        while continue_execution:
            continue_execution = True if address is None else address.contents.value
            try:
                if sys.platform == 'win32':
                    self.obs_shm = mmap.mmap(-1, SIZE, tagname=OBS_SHM_NAME, access=mmap.ACCESS_READ)
                    print("mapped size:", self.obs_shm.size())
                else:
                    self.obs_shm = shared_memory.SharedMemory(name=OBS_SHM_NAME)
                    print("mapped size:", self.obs_shm.size)
            except (FileNotFoundError, ValueError, OSError):
                continue
            except KeyboardInterrupt:
                return
            break

        print(f"Shared memory found")
        if sys.platform == 'win32':
            buf = memoryview(self.obs_shm)
        else:
            buf = memoryview(self.obs_shm.buf)
        while continue_execution:
            try:
                continue_execution = True if address is None else address.contents.value
                image = read_shared_frame(buf, HEADER_SIZE, HEADER_FORMAT)
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
                log_file.close()
                return

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
                    outputs = self.draw.process(result)
                    self.display_card(outputs)

            except KeyboardInterrupt:
                print("Stopping Draw2...")
                close_shared_memory(self.obs_shm)
                close_shared_memory(self.python_shm)
                log_file.close()
                return

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
            self.send_image_to_obs(image)

        for label, count in self.counts.items():
            if count > 20:
                if label not in self.displayed:
                    if time.time() - self.displayed_time > self.minimum_screen_time:
                        self.displayed[label] = time.time()
                        self.displayed_time = time.time()
                        image = self.draw.dataset[int(self.draw.label2id[label])]["image"]
                        self.send_image_to_obs(image)
                    else:
                        if label not in self.queue:
                            self.queue.append(label)
                else:
                    if time.time() - self.displayed[label] > self.minimum_out_of_screen_time:
                        del self.displayed[label]
                        self.counts[label] = 0

    def send_image_to_obs(self, image):
        img_array = np.array(image.convert("RGBA"))
        print(image.size)
        height, width, channels = img_array.shape

        total_size = HEADER_SIZE + img_array.nbytes

        if sys.platform == 'win32' and (self.python_shm is None or total_size != self.python_shm.size()):
            if self.python_shm is not None:
                self.python_shm.close()
            self.python_shm = mmap.mmap(-1, total_size, tagname=PYTHON_SHM_NAME, access=mmap.ACCESS_WRITE)
            self.python_shm.seek(0)
            self.python_shm.write(b"\x00" * total_size)
            self.shm_array = np.ndarray((height, width, channels), dtype=np.uint8, buffer=memoryview(self.python_shm)[HEADER_SIZE:])
        elif self.python_shm is None or total_size != self.python_shm.size:
            if self.python_shm is not None:
                self.python_shm.close()
                self.python_shm.unlink()
            self.python_shm = shared_memory.SharedMemory(name=PYTHON_SHM_NAME, create=True, size=total_size)
            self.shm_array = np.ndarray((height, width, channels), dtype=np.uint8, buffer=self.python_shm.buf[HEADER_SIZE:])
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
        stop_flag=None,
        model_ready=None,
        deck_list="",
        minimum_out_of_screen_time=25,
        minimum_screen_time=6,
        confidence_threshold=5
):
    sh_memory_handler = DrawSharedMemoryHandler(
        deck_list=deck_list,
        minimum_out_of_screen_time=minimum_out_of_screen_time,
        minimum_screen_time=minimum_screen_time,
        confidence_threshold=confidence_threshold
    )

    if stop_flag is not None and model_ready is not None:
        addr = ctypes.cast(ctypes.pythonapi.PyCapsule_GetPointer(stop_flag, b"stop_flag"),
                           ctypes.POINTER(ctypes.c_bool))
        ready_ptr = ctypes.cast(ctypes.pythonapi.PyCapsule_GetPointer(model_ready, b"model_ready"),
                                ctypes.POINTER(ctypes.c_bool))
        print("Running Draw2 with OBS shared memory")
        sh_memory_handler(address=addr, ready_ptr=ready_ptr)
    else:
        print("Running Draw2 without OBS shared memory")
        sh_memory_handler()

    close_shared_memory(sh_memory_handler.obs_shm)
    close_shared_memory(sh_memory_handler.python_shm)


if __name__ == '__main__':
    if sys.platform == 'win32':
        log_dir = os.path.join(os.environ["APPDATA"], "obs-studio")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, "python_subprocess.log")

        # --- logging (flushes immediately) ---
        logging.basicConfig(
            filename=log_path,
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            force=True,  # important if logging was configured before
        )

        # --- redirect print() too ---
        log_file = open(log_path, "a", buffering=1)  # line-buffered
        sys.stdout = log_file
        sys.stderr = log_file

        logging.info("Python subprocess started")
        print("print() is real-time now")

    print("run")
    run()
