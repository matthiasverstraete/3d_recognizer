
import numpy as np

from dataset import Dataset
from .base_camera import Camera


class MockRealsenseCamera(Camera):
    def __init__(
            self,
            name: str,
            mock_dataset: Dataset,
    ):
        super().__init__(name)
        self._mock_dataset = mock_dataset

        if len(self._mock_dataset) == 0:
            raise Exception("Please provide at least 1 mock frame.")

        self._frame_data_index = 0

    def start(self) -> None:
        self._frame_data_index = 0
        super().start()

    def stop(self) -> None:
        super().stop()

    def get(self, timeout_ms=200) -> np.ndarray:
        assert timeout_ms >= 0

        next_frame = self._mock_dataset[self._frame_data_index]
        total = len(self._mock_dataset)
        self._frame_data_index += 1
        self._frame_data_index %= total

        return next_frame[0]
