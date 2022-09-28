from typing import Optional, Callable

import vispy.scene
import numpy as np
from vispy.scene import ArcballCamera
from vispy.util.quaternion import Quaternion


class VispyMarkers(vispy.scene.Markers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data: Optional[np.ndarray] = None

    def set_data(self, pos: Optional[np.ndarray]=None, size: float = 0.01,
                 face_color='white') -> None:
        if pos is None:
            pos_non_optional: np.ndarray = np.array([[0., 0., 0.]])
        else:
            pos_non_optional = pos[np.random.choice(len(pos), size=10000, replace=False), :]

        super().set_data(
            pos=pos_non_optional, edge_width=0.0,
            edge_color=None, face_color=face_color, size=size
        )
        if pos is None:
            self.visible = False
        else:
            self.visible = True
        self._data = pos

    def get_data(self) -> Optional[np.ndarray]:
        return self._data


class IndexedVispyMarkers(VispyMarkers):

    def __init__(self, marker_parent: VispyMarkers, *args, **kwargs):
        self._marker_parent = marker_parent

        super().__init__(*args, **kwargs)

    def set_data(self, pos: Optional[np.ndarray]=None, size: float = 0.01,
                 face_color='white') -> None:

        if pos is None:
            cloud = None
        else:
            cloud = self._marker_parent.get_data()[pos.astype(np.bool)]
        super().set_data(cloud, size, face_color)
        self._data = pos


class VispyView:

    def __init__(
            self, view,
            store_callback: Optional[Callable[[], None]],
            allow_annotation: bool = False,
            offset: np.ndarray = np.array([0, 0, 0.3]),
    ):
        self.view = view
        self._offset = offset
        self._store_callback = store_callback

        self._root_node = vispy.scene.node.Node(name="Root node")
        self.view.add(self._root_node)
        self._point_cloud: VispyMarkers = VispyMarkers(
            parent=self._root_node, scaling=True
        )
        self._point_cloud.set_gl_state("opaque", depth_test=False,
                                       cull_face=False)

        self._annotation = self._vispy_metdata_cloud()
        self._prediction = self._vispy_metdata_cloud()

        self.view.camera = ArcballCamera(fov=0)

        self.view.camera._quaternion = Quaternion(0.707, 0.707, 0.0, 0.0)
        self.view.camera.depth_value = 1.0
        self.view.camera.view_changed()

        if allow_annotation:
            self.view.events.mouse_press.connect(self.viewbox_mouse_event)

    def _vispy_metdata_cloud(self) -> VispyMarkers:
        cloud = IndexedVispyMarkers(
            self._point_cloud, parent=self._root_node, scaling=True
        )
        cloud.set_gl_state("additive")
        return cloud

    @property
    def point_cloud(self) -> VispyMarkers:
        return self._point_cloud

    @point_cloud.setter
    def point_cloud(self, value: np.array) -> None:
        self._point_cloud.set_data(value-self._offset, size=0.001, face_color="red")
        self.annotation = None

    @property
    def annotation(self) -> Optional[np.ndarray]:
        return self._annotation.get_data()

    @annotation.setter
    def annotation(self, value: Optional[np.ndarray]) -> None:
        self._annotation.set_data(value, face_color="blue")

    @property
    def prediction(self) -> Optional[np.ndarray]:
        return self._prediction.get_data()

    @prediction.setter
    def prediction(self, value: np.ndarray) -> None:
        self._prediction.set_data(value, face_color="green")

    def viewbox_mouse_event(self, event):
        if event.button != 3:  # check for modifier keys
            return
        if len(self._point_cloud.get_data()) == 0:
            print("No data captured yet.")
            return

        tform = self.view.scene.transform
        d1 = np.array([0, 0, 1, 0])  # in homogeneous screen coordinates
        point_in_front_of_screen_center = event.pos + d1  # in homogeneous screen coordinates
        p1 = tform.imap(point_in_front_of_screen_center)  # in homogeneous scene coordinates
        p0 = tform.imap(event.pos)  # in homogeneous screen coordinates
        assert (abs(
            p1[3] - 1.0) < 1e-5)  # normalization necessary before subtraction
        assert (abs(p0[3] - 1.0) < 1e-5)
        p0, p1 = p0[:3], p1[:3]

        # check if this intersects with existing annotation
        if self.annotation is not None:
            annotation_cloud = self._point_cloud.get_data()[self.annotation]
            if len(annotation_cloud) > 0:
                lookup = np.where(self.annotation == True)[0]
                d = np.linalg.norm(np.cross(p1 - p0, p0 - annotation_cloud), axis=1)
                min_idx = np.argmin(d)
                if d[min_idx] < 0.01:
                    a = self._annotation.get_data()
                    a[lookup[min_idx]] = False
                    self.annotation = a
                    if self._store_callback is not None:
                        self._store_callback()
                    return

        # if not, add a new point
        d = np.linalg.norm(np.cross(p1 - p0, p0 - self._point_cloud.get_data()), axis=1)
        min_idx = np.argmin(d)
        if self.annotation is None:
            new_annotation = np.zeros(len(self._point_cloud.get_data()), dtype=bool)
        else:
            new_annotation = self.annotation
        new_annotation[min_idx] = True
        self.annotation = new_annotation
        if self._store_callback is not None:
            self._store_callback()
