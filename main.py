import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional

import vispy

from camera import auto_connect_camera
from dataset import Dataset
from predict import Predictor
from train import train_async
from ui import VispyCanvas, DataCapturingFrame, PredictionFrame, TrainFrame

vispy.use("tkinter")


MODELS_PATH = Path("models")
MODELS_PATH.mkdir(parents=True, exist_ok=True)


class Main:

    def __init__(self, window: tk.Tk):
        self.window = window
        window.title("3D gesture capturing")

        main_frame = tk.Frame()
        self._last_timestamp: datetime = datetime.now()
        self.canvas = VispyCanvas(main_frame, self.store_annotation)
        self.canvas.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)

        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(side=tk.BOTTOM)

        self.data_capturing_frame = DataCapturingFrame(
            bottom_frame, self.capture_callback, self.count_captures
        )
        self.data_capturing_frame.pack(side=tk.LEFT)
        self.training_frame = TrainFrame(bottom_frame, self.train, MODELS_PATH)
        self.training_frame.pack(side=tk.LEFT)

        self._prediction_frame = PredictionFrame(
            bottom_frame, self.toggle_prediction, self.set_confidence
        )
        self._prediction_frame.pack(side=tk.RIGHT)
        self._predictor: Optional[Predictor] = None
        self._prediction_interval = 250  # ms
        self._last_prediction = time()

        main_frame.pack(fill=tk.BOTH, expand=True)

        self.camera = auto_connect_camera()
        self.camera.start()

        self.data_capturing_frame.update_count()
        window.bind('<Escape>', self.close)
        window.after(34, self.update_camera_frame)

    def close(self, event):
        progress_tracker = self.training_frame.progress_tracker
        if progress_tracker is not None and \
                progress_tracker.calling_process is not None:
            progress_tracker.calling_process.kill()
            progress_tracker.calling_process.join()
        self.camera.stop()
        self.window.withdraw()  # if you want to bring it back
        sys.exit()  # if you want to exit the entire thing

    def update_camera_frame(self):
        try:
            point_cloud = self.camera.get()
            self.canvas.live_view.point_cloud = point_cloud

            delta = (time() - self._last_prediction)*1000
            if self._predictor is not None and \
                    delta > self._prediction_interval:
                prediction = self._predictor.predict(point_cloud)
                self.canvas.prediction_view.point_cloud = point_cloud
                self.canvas.prediction_view.prediction = prediction
                self._last_prediction = time()
        except Exception as e:
            if str(e) != "No valid frame received.":
                print(e)
                import traceback
                traceback.print_tb(e.__traceback__)

        self.window.after(34, self.update_camera_frame)

    def store_annotation(self) -> None:
        annotation = self.canvas.captured_view.annotation

        now = self._last_timestamp
        dataset_name = self.data_capturing_frame.dataset_name.get()
        dataset = Dataset(Path("data") / dataset_name)
        dataset.set_annotation(now, annotation)

    def capture_callback(self) -> None:
        dataset_name = self.data_capturing_frame.dataset_name.get()
        dataset = Dataset(Path("data") / dataset_name)

        now = datetime.now()

        point_cloud = self.camera.last_cloud
        self.canvas.captured_view.point_cloud = point_cloud
        self._last_timestamp = now
        dataset[now] = point_cloud

    def count_captures(self) -> int:
        dataset_name = self.data_capturing_frame.dataset_name.get()
        dataset = Dataset(Path("data") / dataset_name)
        return len(dataset)

    def train(self) -> None:
        dataset_name = self.data_capturing_frame.dataset_name.get()
        progress_tracker = train_async([Path("data") / dataset_name])
        self.data_capturing_frame.progress_tracker = progress_tracker

    def toggle_prediction(self, enable: bool) -> None:
        if enable:
            current_model_name = self.training_frame.model_name
            if current_model_name == "":
                print("No model loaded yet. First train a model.")
                self._prediction_frame.toggle_predict()
                return
            conf_threshold = self._prediction_frame.confidence_slider.get()
            self._predictor = Predictor(MODELS_PATH / current_model_name,
                                        conf_threshold)
        else:
            self._predictor = None

    def set_confidence(self, value: float) -> None:
        if self._predictor is not None:
            self._predictor.confidence_threshold = float(value)


if __name__ == '__main__':
    window = tk.Tk()
    main = Main(window)
    window.mainloop()
