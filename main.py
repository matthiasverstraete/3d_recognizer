import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from time import time
from tkinter import ttk
from typing import Callable, Optional

import numpy as np
import vispy
import vispy.scene

from camera import auto_connect_camera
from dataset import Dataset
from predict import Predictor
from train import train_async, ProgressTracker
from ui import VispyView

vispy.use("tkinter")


MODELS_PATH = Path("models")
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# TODO: split of all UI components in separate folder


class VispyCanvas(tk.Frame):

    def __init__(self, parent, store_callback):
        super().__init__(parent, height=200, width=400)
        self.pack_propagate(False)
        self._store_callback = store_callback

        self._canvas = vispy.scene.SceneCanvas(
            title="visualization 3D",
            keys="interactive",
            show=True,
            fullscreen=False,
            size=(200, 600),
            position=(0, 0),
            parent=self,
        )
        self._canvas.native.pack(
            side=tk.LEFT,
            anchor="nw",
            fill=tk.BOTH,
            expand=True
        )
        grid = self._canvas.central_widget.add_grid()

        self.live_view = VispyView(
            grid.add_view(border_color=(0.5, 0.5, 0.5, 1), row=0, col=0),
            store_callback, offset=np.array([0, 0, 0])
        )
        self.captured_view = VispyView(
            grid.add_view(border_color=(0.5, 0.5, 0.5, 1), row=0, col=1),
            store_callback,
            allow_annotation=True,
            offset=np.array([0, 0, 0])
        )
        self.prediction_view = VispyView(
            grid.add_view(border_color=(0.5, 0.5, 0.5, 1), row=0, col=2),
            store_callback,
            offset=np.array([0, 0, 0])
        )

        self.live_view.view.camera.link(self.captured_view.view.camera)
        self.live_view.view.camera.link(self.prediction_view.view.camera)


class DataCapturingFrame(tk.Frame):

    def __init__(
            self, master, store_capture: Callable,
            count_captures: Callable, train: Callable):
        super().__init__(master)

        self._count_captures = count_captures
        self._train = train
        self.dataset_name_label = tk.Label(self, anchor="e",
                                           text="Dataset name:")
        self.dataset_name_label.grid(row=0, column=0)
        self.dataset_name = tk.Entry(self)
        self.dataset_name.bind("<KeyRelease>", self.update_count)
        self.dataset_name.grid(row=0, column=1, sticky=tk.EW)

        self._store_capture = store_capture
        self.capture = tk.Button(self, anchor="e", text="Capture",
                                 command=self.capture_callback)
        self.capture.grid(row=2, column=0, columnspan=2, sticky=tk.EW)

        self.counter = tk.Label(self)
        self.counter.grid(row=3, column=0, columnspan=2)

        self._train_button = tk.Button(self, anchor="e", text="Train",
                                       command=self.start_training)
        self._train_button.grid(row=4, column=0, columnspan=2)
        self._progress_bar = ttk.Progressbar(
            self, orient=tk.HORIZONTAL, length=100, mode="determinate"
        )
        self._progress_bar.grid(row=5, column=0, columnspan=2)

        self._progress_tracker: Optional[ProgressTracker] = None

        self._model_label = tk.Label(self, anchor="e", text="Model: ")
        self._model_label.grid(row=6, column=0)
        self._model_name = tk.Label(self, anchor="e", text="")
        self._model_name.grid(row=6, column=1)
        self.update_model_name()

    def capture_callback(self):
        self._store_capture()
        self.update_count()

    def update_count(self, *args) -> bool:
        self.counter['text'] = self._count_captures()
        return True

    def update_model_name(self) -> None:
        all_models = sorted(MODELS_PATH.iterdir())
        if len(all_models) > 0:
            latest_model = all_models[-1]
            self._model_name["text"] = latest_model.name

    def start_training(self) -> None:
        self._train_button["state"] = "disabled"
        self._progress_bar["value"] = 1
        self._train()

    def do_progress_check(self) -> None:
        if self._progress_tracker is None:
            return

        progress = self._progress_tracker.check_progress()
        self._progress_bar['value'] = progress

        if progress != 100:
            self.after(500, self.do_progress_check)
        else:
            self._train_button["state"] = "active"
            self._progress_tracker = None
            self.update_model_name()

    @property
    def progress_tracker(self) -> Optional[ProgressTracker]:
        return self._progress_tracker

    @progress_tracker.setter
    def progress_tracker(self, value: Optional[ProgressTracker]) -> None:
        self._progress_tracker = value
        if value is not None:
            self.after(500, self.do_progress_check)


class PredictionFrame(tk.Frame):

    def __init__(self, master, toggle_predict, set_confidence):
        super().__init__(master)
        self._toggle_predict = toggle_predict

        tk.Label(self, text="Confidence").pack()
        self.confidence_slider = tk.Scale(self, from_=0, to=1, resolution=0.01,
                                          command=set_confidence)
        self.confidence_slider.set(0.5)
        self.confidence_slider.pack()

        self._predict_button = tk.Button(self, anchor="e", text="Predict",
                                         command=self.toggle_predict)
        self._predict_button.pack(side=tk.BOTTOM)

    def toggle_predict(self) -> None:
        if self._predict_button.config('relief')[-1] == 'sunken':
            self._predict_button.config(relief="raised")
            self._toggle_predict(False)
        else:
            self._predict_button.config(relief="sunken")
            self._toggle_predict(True)


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
            bottom_frame, self.capture_callback, self.count_captures,
            self.train
        )
        self.data_capturing_frame.pack(side=tk.LEFT)

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
        if self.data_capturing_frame.progress_tracker is not None and \
            self.data_capturing_frame.progress_tracker.calling_process is not None:
            self.data_capturing_frame.progress_tracker.calling_process.kill()
            self.data_capturing_frame.progress_tracker.calling_process.join()
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
            current_model_name = self.data_capturing_frame._model_name["text"]
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
