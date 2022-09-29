from pathlib import Path

import numpy as np
import vispy
import vispy.scene
import vispy.app

from dataset import Dataset, DatasetMerged
from randlanet import Model
from ui import VispyView, Label

vispy.use("tkinter")

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


def visualize(
        point_cloud: np.ndarray,
        annotation: np.ndarray,
        prediction: np.ndarray
) -> bool:
    do_break = False
    canvas = vispy.scene.SceneCanvas(
        title="visualization 3D",
        keys="interactive",
        show=True,
        fullscreen=False,
        size=(1000, 600),
        position=(0, 0),
    )

    def process_key(event):
        nonlocal do_break
        if event.key == vispy.keys.ESCAPE:
            do_break = True
            vispy.app.quit()
        elif event.key == vispy.keys.ENTER:
            vispy.app.quit()

    canvas.events.key_press.connect(process_key)

    view = canvas.central_widget.add_view()
    vispy_view = VispyView(view, None)
    vispy_view.point_cloud = point_cloud
    vispy_view.annotation = annotation
    vispy_view.prediction = prediction

    help_text = (
        "red: captured data \n"
        "green: prediction from the model \n"
        "blue: annotation \n"
        "white: overlap of all three above \n"
        "Press enter for next sample. Press escape to stop."
    )
    help_label = Label(
        help_text, color="white", anchor_x="left", anchor_y="bottom",
    )
    canvas.central_widget.add_widget(help_label)

    canvas.show(visible=True)

    vispy.app.run()

    return do_break


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "Predictor",
        usage="python3 predict.py -m models/2022_09_20__08_13_58_478586000 "
              "-d data/dataset1",
        description="This script allows visualizing a prediction without a UI."
        "The script will iterate over each sample in the dataset(s) "
        "and visualize it one by one. Press 'enter' to go to the next sample. "
        "Press 'escape' to stop predictions.",
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="Select the model which should be used. Path should be relative "
             "to main project directory.")
    parser.add_argument(
        "-d", "--dataset", nargs="+", required=True,
        help="Select one or multiple datasets to predict. "
             "Paths should be relative to main project directory.",
    )
    parser.add_argument(
        "-c", "--confidence", required=False, default=0.5, type=float,
        help="Choose which confidence threshold to use. default value is 0.5"
    )
    args = parser.parse_args()
    project_dir = Path(__file__).absolute().parent

    predictor = Predictor(project_dir / args.model,
                          confidence_threshold=args.confidence)
    datasets = [
        Dataset(project_dir / dataset_name, only_annotated=False,
                broaden_annotations=True)
        for dataset_name in args.dataset
    ]
    dataset = DatasetMerged(datasets)

    for (point_cloud, features, annotation) in dataset:
        prediction = predictor.predict(point_cloud)
        do_break = visualize(point_cloud, annotation, prediction)
        if do_break:
            break
