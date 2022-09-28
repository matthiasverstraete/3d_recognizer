import tkinter as tk
from typing import Callable


class DataCapturingFrame(tk.Frame):

    def __init__(
            self, master, store_capture: Callable,
            count_captures: Callable):
        super().__init__(master)

        self._count_captures = count_captures
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

    def capture_callback(self):
        self._store_capture()
        self.update_count()

    def update_count(self, *args) -> bool:
        self.counter['text'] = self._count_captures()
        return True