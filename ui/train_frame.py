import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Callable, Optional

from train import ProgressTracker


class TrainFrame(tk.Frame):

    def __init__(
            self, master, train_callback: Callable, models_path: Path):
        super().__init__(master)

        self._train_callback = train_callback

        self._train_button = tk.Button(self, anchor="e", text="Train",
                                       command=self.start_training)
        self._train_button.grid(row=4, column=0, columnspan=2)
        self._progress_bar = ttk.Progressbar(
            self, orient=tk.HORIZONTAL, length=100, mode="determinate"
        )
        self._progress_bar.grid(row=5, column=0, columnspan=2)

        self._progress_tracker: Optional[ProgressTracker] = None

        self._models_path = models_path
        self._model_label = tk.Label(self, anchor="e", text="Model: ")
        self._model_label.grid(row=6, column=0)
        self._model_name = tk.Label(self, anchor="e", text="")
        self._model_name.grid(row=6, column=1)
        self.update_model_name()

    @property
    def model_name(self) -> str:
        return self._model_name["text"]

    def update_model_name(self) -> None:
        all_models = sorted(self._models_path.iterdir())
        if len(all_models) > 0:
            latest_model = all_models[-1]
            self._model_name["text"] = latest_model.name

    def start_training(self) -> None:
        self._train_button["state"] = "disabled"
        self._progress_bar["value"] = 1
        self._train_callback()

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
