import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from time import time
from tkinter import ttk
from typing import Callable, Optional

import cv2
from PIL import Image, ImageTk
import numpy as np
import vispy
import vispy.scene
from vispy.scene import ArcballCamera
from vispy.util.quaternion import Quaternion

from dataset import Dataset
from predict import Predictor
from realsense_camera import RealsenseCamera
from train import train_async, ProgressTracker

classes = ("thumbs up", "thumbs down")

vispy.use("tkinter")


MODELS_PATH = Path("models")
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# TODO: add option to run with mock camera
# TODO: split of all UI components in separate folder

class VispyView:

    def __init__(self, view, store_callback,  allow_annotation: bool = False):
        self.view = view
        self._store_callback = store_callback

        self._root_node = vispy.scene.node.Node(name="Root node")
        self.view.add(self._root_node)
        self._point_cloud = vispy.scene.Markers(
            parent=self._root_node, scaling=True
        )
        self._point_cloud.set_gl_state("opaque", depth_test=False,
                                       cull_face=False)
        self._point_cloud_data = np.array([])

        self._annotation = vispy.scene.Markers(
            parent=self._root_node, scaling=True
        )
        self._annotation.set_gl_state("additive")
        self._annotation_data: Optional[np.ndarray] = None

        self.view.camera = ArcballCamera()

        self.view.camera.fov = 0.0
        self.view.camera._quaternion = Quaternion(0.707, 0.707, 0.0, 0.0)
        self.view.camera.depth_value = 2.0
        self.view.camera.view_changed()

        if allow_annotation:
            self.view.events.mouse_press.connect(self.viewbox_mouse_event)

    @property
    def point_cloud(self) -> vispy.scene.Markers:
        return self._point_cloud

    @point_cloud.setter
    def point_cloud(self, value: np.array) -> None:
        self._point_cloud.set_data(
            value, edge_width=0.0, edge_color=None,
            face_color="red", size=0.005
        )
        self._point_cloud.visible = True

        self._point_cloud_data = value
        self.annotation = None

    @property
    def annotation(self) -> np.array:
        return self._annotation_data

    @annotation.setter
    def annotation(self, value: np.ndarray) -> None:
        if value is None:
            self._annotation.set_data(
                np.array([[0, 0, 0]]), edge_width=0.0,
                edge_color=None, face_color="blue", size=0.01
            )
            self._annotation.visible = False
        else:
            self._annotation.set_data(
                self._point_cloud_data[value], edge_width=0.0,
                edge_color=None, face_color="blue", size=0.01
            )
            self._annotation.visible = True
        self._annotation_data = value

    def viewbox_mouse_event(self, event):
        if event.button != 1:  # check for modifier keys
            return

        tform = self.view.scene.transform
        d1 = np.array([0, 0, 1, 0])  # in homogeneous screen coordinates
        point_in_front_of_screen_center = event.pos + d1  # in homogeneous screen coordinates
        p1 = tform.imap(point_in_front_of_screen_center)  # in homogeneous scene coordinates
        p0 = tform.imap(event.pos)  # in homogeneous screen coordinates
        assert (abs(
            p1[3] - 1.0) < 1e-5)  # normalization necessary before subtraction
        assert (abs(p0[3] - 1.0) < 1e-5)
        p0, p1 = p0[:3], p1[:3]

        # check if this intersects with existing annotation
        if self.annotation is not None:
            annotation_cloud = self._point_cloud_data[self.annotation]
            if len(annotation_cloud) > 0:
                lookup = np.where(self.annotation == True)[0]
                d = np.linalg.norm(np.cross(p1 - p0, p0 - annotation_cloud), axis=1)
                min_idx = np.argmin(d)
                if d[min_idx] < 0.01:
                    a = self._annotation_data
                    a[lookup[min_idx]] = False
                    self.annotation = a
                    self._store_callback()
                    return

        # if not, add a new point
        d = np.linalg.norm(np.cross(p1 - p0, p0 - self._point_cloud_data), axis=1)
        min_idx = np.argmin(d)
        if self.annotation is None:
            new_annotation = np.zeros(len(self._point_cloud_data), dtype=bool)
        else:
            new_annotation = self.annotation
        new_annotation[min_idx] = True
        self.annotation = new_annotation
        self._store_callback()


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
            store_callback
        )
        self.captured_view = VispyView(
            grid.add_view(border_color=(0.5, 0.5, 0.5, 1), row=0, col=1),
            store_callback,
            allow_annotation=True
        )
        self.prediction_view = VispyView(
            grid.add_view(border_color=(0.5, 0.5, 0.5, 1), row=0, col=2),
            store_callback
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
        latest_model = sorted(MODELS_PATH.iterdir())[-1]
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

    def __init__(self, master, toggle_predict):
        super().__init__(master)
        self._toggle_predict = toggle_predict

        self.class_sliders = {}
        for class_name in classes:
            self.class_sliders[class_name] = tk.Scale(self, from_=0, to=1, resolution=0.01)
            self.class_sliders[class_name].pack(side=tk.LEFT)

        self._predict_button = tk.Button(self, anchor="e", text="Predict",
                                         command=self.toggle_predict)
        self._predict_button.pack(side=tk.BOTTOM)

    def toggle_predict(self):
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

        prediction_frame = PredictionFrame(bottom_frame, self.toggle_prediction)
        prediction_frame.pack(side=tk.RIGHT)
        self._predictor: Optional[Predictor] = None
        self._prediction_interval = 250  # ms
        self._last_prediction = time()

        main_frame.pack(fill=tk.BOTH, expand=True)

        self.camera = RealsenseCamera.auto_connect()
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
                self.canvas.prediction_view.annotation = prediction
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
            self._predictor = Predictor(MODELS_PATH / current_model_name)
        else:
            self._predictor = None


if __name__ == '__main__':
    window = tk.Tk()
    main = Main(window)
    window.mainloop()
