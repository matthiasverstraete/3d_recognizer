import numpy as np


class Camera:
    def __init__(
            self,
            name: str,
    ):
        self.name = name
        self._running = False
        self._last_cloud = np.array([])

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def device_connected(self) -> bool:
        return True

    def get(self, timeout_ms=200) -> np.ndarray:
        raise NotImplementedError()

    @property
    def last_cloud(self) -> np.ndarray:
        return self._last_cloud
