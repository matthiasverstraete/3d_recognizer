from pathlib import Path

import numpy as np

from randlanet import Model


class Predictor:

    def __init__(
            self, model_path: Path, confidence_threshold: float = 0.5
    ) -> None:
        self._model = Model.load(model_path, use_gpu=True)
        self.confidence_threshold: float = confidence_threshold

        # first prediction is always much slower. So do one as a warmup
        dummy_cloud = np.random.random((30, 3))
        self._model.predict(dummy_cloud)

    def predict(self, point_cloud: np.ndarray) -> np.ndarray:
        confidences_all_classes = self._model.predict(point_cloud)
        confidences = confidences_all_classes[1, :]  # class 0 is background
        prediction_mask = confidences > self.confidence_threshold

        return prediction_mask


if __name__ == '__main__':
    # TODO: include help
    predictor = Predictor(Path("models/2022_07_26__10_13_30_365682000"))
    prediction = predictor.predict(np.zeros((13, 3), dtype=np.uint8))
    print(prediction)
    # TODO: visualize predictions nicely
