from datetime import datetime
from multiprocessing import Queue, Process, set_start_method
from pathlib import Path
from queue import Empty
from typing import Optional, List

import tensorboard

from dataset import Dataset, DatasetMerged
from randlanet import Model, RandLANetSettings, TrainingSettings, \
    AugmentationSettings


class ProgressTracker:

    def __init__(self, queue: Queue):
        self._queue = queue
        self.calling_process: Optional[Process] = None

        self.progress_cache: int = 1

    def set_progress(self, value: int) -> None:
        self._queue.put(value)

    def check_progress(self) -> int:
        last_progress = self.progress_cache
        while True:
            try:
                last_progress = self._queue.get_nowait()
            except Empty:
                break

        if last_progress == 100:
            if self.calling_process is not None:
                if self.calling_process.is_alive():
                    last_progress = 99

        if self.calling_process is None or not self.calling_process.is_alive():
            last_progress = 100

        self.progress_cache = last_progress
        return last_progress


def train(dataset_names: List[Path], tracker: Optional[ProgressTracker] = None):
    datasets = [Dataset(dataset_name, broaden_annotations=True) for dataset_name in dataset_names]
    dataset_merged = DatasetMerged(datasets)
    train_dataset, validation_dataset = dataset_merged.split()

    settings = RandLANetSettings(n_classes=2, n_features=0, knn="naive",
                                 n_points=2500, n_neighbors=32, decimation=4)
    model = Model(settings, use_gpu=True)

    training_settings = TrainingSettings(
        epochs=50,
        batch_size=4,
        learning_rate=1e-2,
        early_stopping=False,
    )

    augmentation_settings = AugmentationSettings(
        jitter_variance=0.01,
        jitter_limit=0.05,
        scale_limit=0.2,
        shift_limit=0.1,
        rotation_angle_variances=(0.06, 0.06, 0.06),
        rotation_angle_limits=(0.18, 0.18, 0.18),
    )

    # init tensorboard

    now = datetime.now()
    now_str = "%04.i_%02.i_%02.i__%02.i_%02.i_%02.i_%06.i000" % (
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
        now.microsecond,
    )
    log_dir = Path(f"training_log/{now_str}")
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, "--logdir", "training_log"])
    tb.launch()

    def progress_callback(epoch, metrics):
        if tracker is not None:
            tracker.set_progress(int(100*epoch/training_settings.epochs))

    print(f"\nStarting training and logging at {log_dir} ...")
    print(f"Training settings are: {training_settings}")
    print(f"Augmentation settings are: {augmentation_settings}\n")
    model.train(
        train_dataset,
        validation_dataset,
        training_settings,
        augmentation_settings,
        log_dir,
        ["background", "fingerpoint"],
        callbacks=[progress_callback]
    )
    model_path = Path("models") / now_str
    model.save(model_path)
    print(f"\nModel saved to {model_path}")


def train_async(dataset_names: List[Path]) -> ProgressTracker:
    set_start_method('spawn')
    tracker = ProgressTracker(Queue())
    p = Process(target=train, args=(dataset_names, tracker))
    p.start()
    tracker.calling_process = p

    return tracker


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "Trainer",
        description="Use this script to train a model without the UI. This "
                    "script also allows training on multiple datasets by "
                    "combining them in one large dataset.",
        usage="python3 train.py -d data/dataset1 data/dataset2"
    )
    parser.add_argument(
        "-d", "--dataset", nargs="+",
        help="Select one or multiple datasets to train on. "
        "Paths should be relative to main project directory",
        required=True
    )
    args = parser.parse_args()
    project_dir = Path(__file__).absolute().parent

    from time import sleep, time
    start = time()
    datasets = [project_dir / Path(path) for path in args.dataset]
    tracker = train_async(datasets)

    while True:
        progress = tracker.check_progress()
        print(progress)
        if progress == 100:
            break
        sleep(1)
    print(f"training took {(time()-start)/60} seconds")