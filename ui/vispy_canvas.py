import tkinter as tk
import vispy.scene
import numpy as np
from .vispy_view import VispyView


class VispyCanvas(tk.Frame):
    """
    This vispy based canvas will visualize point clouds in 3 separate windows.
    One for live viewing, one for annotation and one for prediction
    """

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
