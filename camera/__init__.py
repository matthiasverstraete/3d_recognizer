from pathlib import Path

from dataset import Dataset
from .realsense_camera import RealsenseCamera
from .mock_camera import MockRealsenseCamera
from .base_camera import Camera

import pyrealsense2.pyrealsense2 as rs


def auto_connect_camera() -> Camera:
    context = rs.context()

    for device in context.query_devices():
        if device.get_info(rs.camera_info.name) == "Intel RealSense L515":
            serial = device.get_info(rs.camera_info.serial_number)
            return RealsenseCamera(serial, serial)

    return MockRealsenseCamera("mock", Dataset(
        Path("__file__").parent / "data" / "mock", only_annotated=False
    ))


__all__ = [
    "Camera",
    "auto_connect_camera"
]