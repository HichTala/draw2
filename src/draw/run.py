import ctypes
import struct
import time
from collections import deque
from multiprocessing import shared_memory

from draw.draw import Draw
from draw.utils import read_shared_frame, get_deck_list

import logging

logging.disable(logging.CRITICAL)

OBS_SHM_NAME = "obs_shared_memory"
PYTHON_SHM_NAME = "python_shared_memory"
HEADER_FORMAT = "II"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class DrawSharedMemoryHandler:
    def __init__(self, deck_list="", minimum_out_of_screen_time=25, minimum_screen_time=6, confidence_threshold=5):
        self.draw = Draw(deck_lists=get_deck_list(deck_list), confidence_threshold=confidence_threshold)

        self.obs_shm = None
        self.python_shm = None
        self.buf = None

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
                self.obs_shm = shared_memory.SharedMemory(name=OBS_SHM_NAME)
            except FileNotFoundError:
                continue
            except ValueError:
                continue
            except KeyboardInterrupt:
                return
            break

        print(f"Shared memory found")
        while continue_execution:
            try:
                continue_execution = True if address is None else address.contents.value
                image = read_shared_frame(self.obs_shm.buf, HEADER_SIZE, HEADER_FORMAT)
            except TypeError:
                breakpoint()
            except ValueError:
                breakpoint()
            except KeyboardInterrupt:
                print("Stopping Draw2...")
                self.obs_shm.close()
                self.python_shm.close()
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
            # except Exception as e:
            #     print(f"Error processing shared memory: {e}")
            #     breakpoint()
            except KeyboardInterrupt:
                print("Stopping Draw2...")
                self.obs_shm.close()
                self.python_shm.close()
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
        img_rgba = image.convert("RGBA")
        width, height = img_rgba.size
        img_bytes = img_rgba.tobytes()

        total_size = HEADER_SIZE + len(img_bytes)

        if self.python_shm is None:
            self.python_shm = shared_memory.SharedMemory(name=PYTHON_SHM_NAME, create=True, size=total_size)

        if total_size != self.python_shm.size:
            self.python_shm.close()
            self.python_shm.unlink()
            self.python_shm = shared_memory.SharedMemory(name=PYTHON_SHM_NAME, create=True, size=total_size)

        self.python_shm.buf[:HEADER_SIZE] = struct.pack(HEADER_FORMAT, width, height)
        self.python_shm.buf[HEADER_SIZE:] = img_bytes

        print(f"Sent image {width}x{height} to OBS")


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

    if sh_memory_handler.obs_shm is not None:
        sh_memory_handler.obs_shm.unlink()
    if sh_memory_handler.python_shm is not None:
        sh_memory_handler.python_shm.unlink()


if __name__ == '__main__':
    run()
