from multiprocessing import Process, JoinableQueue, Value
from pathlib import Path
import cv2
import pyrealsense2 as rs
import numpy as np
import time

class Camera:
    """
    Interface for using Intel RealSense D435.
    Adapted from camera.py in /erl_xArm/devices/camera.py
    """

    def __init__(self, write_path="camera_imgs/", fps=60):
        self.write_path = write_path
        self.fps = fps
        self.capture_process = None
        self.write_process = None
        self.pipeline = rs.pipeline()

        self.queue = JoinableQueue()
        self.config = rs.config()
        self.alive = False

        self.age_sum = Value('d', 0.0)

        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)

        self.pipeline.start(self.config)
        Path(self.write_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to {Path(self.write_path).absolute()}.")
        # self.write_process = Process(target=self._save_frames, daemon=True)
        # self.write_process.start()
        time.sleep(4)   # ensure pipeline is ready before capturing frames

    def flush(self):
        for _ in range(self.fps):
            self.pipeline.wait_for_frames(timeout_ms=5000)

    def __del__(self):
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
        print("Camera pipeline stopped.", flush=True)
        # print(f"Waiting for {self.queue.qsize()}", flush=True)
        # self.queue.join()
        # print("Frames saved.", flush=True)

    def _save_frame(self, frame):
        """Saves a single frame to disk."""
        img = np.asanyarray(frame.get_data())
        timestamp = frame.get_timestamp()
        path = Path(f"{self.write_path}/frame_{timestamp}.jpg")
        cv2.imwrite(path, img)

    def _save_frames(self):
        """Saves any frames in the queue to disk while pipeline is active."""
        Path(self.write_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to {Path(self.write_path).absolute()}.")
        while True:
            try:
                frame = self.queue.get(timeout=5)
            except Exception as e:
                print(e)
                print(self.queue.qsize())
                continue
            self._save_frame(frame)

    def _get_frame(self):
        """Enqueues the latest frame."""
        start = time.time()
        success, frames = self.pipeline.try_wait_for_frames()
        if not success:
            print(f"Failed to get frame at time\t{start}")
            return None
        color_frame = frames.get_color_frame()
        print(f"Got frame at time\t{start}")
        # self._save_frame(color_frame)
        return np.asanyarray(color_frame.get_data())

    def get_frame(self):
        """Parallelized wrapper for _get_frame."""
        self.capture_process = Process(target=self._get_frame, daemon=True)
        self.capture_process.start()
