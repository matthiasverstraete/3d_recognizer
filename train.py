from datetime import datetime
from multiprocessing import Queue, Process, set_start_method
from pathlib import Path
from queue import Empty
from typing import Optional

import tensorboard

from dataset import Dataset
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


def train(dataset_name: Path, tracker: Optional[ProgressTracker] = None):
    dataset = Dataset(dataset_name)
    train_dataset, validation_dataset = dataset.split()

    settings = RandLANetSettings(n_classes=2, n_features=0, knn="naive")
    model = Model(settings, use_gpu=True)

    training_settings = TrainingSettings(
        epochs=50,
        batch_size=2,
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


def train_async(dataset_name: Path) -> ProgressTracker:
    set_start_method('spawn')
    tracker = ProgressTracker(Queue())
    p = Process(target=train, args=(dataset_name, tracker))
    p.start()
    tracker.calling_process = p

    return tracker


if __name__ == '__main__':
    from time import sleep
    tracker = train_async(Path("data/matthias_office"))
    while True:
        progress = tracker.check_progress()
        print(progress)
        if progress == 100:
            break
        sleep(1)
