import numpy as np
from pyrealsense2 import pyrealsense2 as rs

from camera.base_camera import Camera


class RealsenseCamera(Camera):
    def __init__(
            self,
            name: str,
            device_serial: str,
    ):
        super().__init__(name)
        self._realsense_config = rs.config()

        self._context = rs.context()
        self._pipeline = rs.pipeline(self._context)
        self._serial = device_serial

        self._realsense_config.enable_device(device_serial)

        self._realsense_config.enable_stream(
            rs.stream.depth, 1024, 768, rs.format.z16, 30
        )

        self._temporal_filter = rs.temporal_filter(
            0.33, 100, 0,
        )
        self.pc_process = rs.pointcloud()

        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)

        self._pipeline_profile = self._realsense_config.resolve(
            pipeline_wrapper
        )

        device = self._pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        if device_product_line != "L500":
            raise Exception("Expected an L515 camera")

    @property
    def serial(self) -> str:
        return self._serial

    @staticmethod
    def _validate_point_cloud(point_cloud) -> bool:
        non_zeros = point_cloud[~np.all(point_cloud == 0.0, axis=1)]

        z_mean = np.mean(non_zeros[:, 2])
        return z_mean < 2

    def _configure_device(self) -> None:
        device = self._pipeline_profile.get_device()
        depth_sensor: rs.sensor = device.first_depth_sensor()

        # depth sensor settings
        depth_sensor.set_option(rs.option.min_distance, 0)
        depth_sensor.set_option(rs.option.digital_gain, 1.0)
        depth_sensor.set_option(rs.option.laser_power, 100)
        depth_sensor.set_option(rs.option.receiver_gain, 9)
        depth_sensor.set_option(rs.option.noise_filtering, 6)

    def start(self) -> None:
        """
        Start the pipeline. This means the camera will start capturing from all
        enabled streams. Be warned that lidar streams will start sending out
        lasers from now.
        """
        if self._running:
            return

        self._configure_device()

        self._pipeline.start(self._realsense_config)
        super().start()

    def stop(self) -> None:
        """
        Stop the pipeline. All streams are stopped and lidar sensors stop
        sending out lasers.
        """
        if not self._running:
            return

        self._pipeline.stop()
        super().stop()

    def get(
            self, timeout_ms: int = 200
    ) -> np.ndarray:
        """
        Get the latest point cloud from the device. If no point cloud can be
        fetched within timeout_ms, it will raise an error.
        :return: The point cloud
        """
        if not self._running:
            raise Exception(
                f"Realsense pipeline is not running."
            )
        success, frames = self._pipeline.try_wait_for_frames(
            timeout_ms=timeout_ms
        )
        if not success:
            raise Exception("No frame received.")

        depth_frame = frames.get_depth_frame()

        if self._temporal_filter is not None:
            depth_frame = self._temporal_filter.process(depth_frame)
        points_data = self.pc_process.calculate(depth_frame)
        points = np.asanyarray(points_data.get_vertices())\
            .view(np.float32).reshape(-1, 3)

        if not self._validate_point_cloud(points):
            raise Exception("No valid frame received.")

        # filter z
        points1 = points[points[:, 2] < 0.6]
        points2 = points1[0.001 < points1[:, 2]]

        self._last_cloud = points2

        return points2
