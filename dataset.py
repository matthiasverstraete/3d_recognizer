from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple, overload, Union, Optional, List

import numpy as np


def annotation_to_cloud(point_cloud: np.ndarray,
                        annotation: np.ndarray,
                        radius: float = 0.005) -> np.ndarray:
    output = []

    for annotation_point in annotation:
        ds = np.abs(np.linalg.norm(annotation_point - point_cloud, axis=1))
        output.append(ds < radius)

    return np.logical_or.reduce(output).astype(np.uint8)


class Dataset(Sequence):

    def __init__(self, root_path: Path, only_annotated: bool = True,
                 selection: Optional[List[int]] = None):
        self._root_path = root_path
        self._only_annotated = only_annotated
        self._selection = selection

    def __len__(self):
        if self._selection is not None:
            return len(self._selection)
        if self._only_annotated:
            return len([a for a in self._root_path.glob("*_annotation*") if a.is_file()])
        return len([a for a in self._root_path.glob("*_data*") if a.is_file()])

    def _get_item_index(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = sorted(self._root_path.glob("*_data*"))

        while True:
            if self._selection is not None:
                selected_sample_path = data[self._selection[index]]
            else:
                selected_sample_path = data[index]

            selected_sample = selected_sample_path.name.split("_data")[0]

            try:
                return self._get_item_str(selected_sample)
            except Exception as e:
                if str(e) != "No annotation":
                    raise
                index += 1

    def _get_item_datetime(
            self, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._get_item_str(Dataset.timestamp(timestamp))

    def _get_item_str(
            self, index: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        item_path = self._root_path / (index + "_data.npy")
        if not item_path.exists():
            raise Exception(f"index {index} doesn't exist in dataset.")

        point_cloud = np.load(str(item_path))
        annotation_path = self._root_path / (index + "_annotation.npy")
        if annotation_path.exists():
            annotation = np.load(str(annotation_path))

            annotation_cloud = annotation_to_cloud(point_cloud, annotation)
        else:
            if self._only_annotated:
                raise Exception("No annotation")
            annotation_cloud = np.zeros([point_cloud.shape[0],], dtype=np.uint8)

        return point_cloud, np.zeros((point_cloud.shape[0], 0)), annotation_cloud

    def __getitem__(
            self, index: Union[int, datetime]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(index, datetime):
            return self._get_item_datetime(index)
        elif isinstance(index, int):
            return self._get_item_index(index)
        else:
            raise Exception("invalid key!")

    def __setitem__(self, index: datetime, value: np.ndarray) -> None:
        root = self._root_path / Dataset.timestamp(index)

        self._root_path.mkdir(parents=True, exist_ok=True)
        np.save(str(root) + "_data", value)

    def set_annotation(self, index: datetime, value: np.ndarray) -> None:
        root = self._root_path / Dataset.timestamp(index)
        self._root_path.mkdir(parents=True, exist_ok=True)
        np.save(str(root) + "_annotation", value)

    @classmethod
    def timestamp(cls, time: Optional[datetime]) -> str:
        input_datetime: datetime = datetime.now()
        if time is not None:
            input_datetime = time
        return "%04.i_%02.i_%02.i__%02.i_%02.i_%02.i_%06.i000" % (
            input_datetime.year,
            input_datetime.month,
            input_datetime.day,
            input_datetime.hour,
            input_datetime.minute,
            input_datetime.second,
            input_datetime.microsecond,
        )

    def split(self, percentage: float = 0.8) -> "Tuple[Dataset, Dataset]":
        indices = list(range(len(self)))
        split_index = int(percentage*len(indices))
        return Dataset(
            self._root_path, self._only_annotated, selection=indices[: split_index]
        ), Dataset(
            self._root_path, self._only_annotated, selection=indices[split_index:]
        )

