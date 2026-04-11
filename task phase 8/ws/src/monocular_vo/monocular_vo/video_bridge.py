from __future__ import annotations

import copy
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from monocular_vo.io import build_approx_camera_info, load_camera_info_from_yaml


class VideoBridge(Node):
    def __init__(self) -> None:
        super().__init__("video_bridge")

        self.declare_parameter("video_path", "/project/video.mp4")
        self.declare_parameter("output_width", 1280)
        self.declare_parameter("output_height", 720)
        self.declare_parameter("publish_fps", 15.0)
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("camera_info_mode", "approx")
        self.declare_parameter("horizontal_fov_deg", 90.0)
        self.declare_parameter("camera_info_yaml", "")

        self.video_path = self.get_parameter("video_path").get_parameter_value().string_value
        self.output_width = int(self.get_parameter("output_width").value)
        self.output_height = int(self.get_parameter("output_height").value)
        self.publish_fps = float(self.get_parameter("publish_fps").value)
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.camera_info_mode = self.get_parameter("camera_info_mode").get_parameter_value().string_value
        self.horizontal_fov_deg = float(self.get_parameter("horizontal_fov_deg").value)
        self.camera_info_yaml = self.get_parameter("camera_info_yaml").get_parameter_value().string_value

        if self.output_width <= 0 or self.output_height <= 0:
            raise RuntimeError("output_width and output_height must be positive.")
        if self.publish_fps <= 0.0:
            raise RuntimeError("publish_fps must be positive.")

        video_path = Path(self.video_path)
        if not video_path.is_file():
            raise RuntimeError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.source_fps <= 0.0:
            self.source_fps = self.publish_fps

        self.source_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_stride = max(1, int(round(self.source_fps / self.publish_fps)))

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, self.image_topic, qos_profile_sensor_data)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, qos_profile_sensor_data)

        self.camera_info_template = self._build_camera_info_template()
        self.replay_start_time = None
        self.current_source_frame_index = 0
        self.next_source_frame_index = 0
        self.frames_published = 0

        self.get_logger().info(
            (
                f"Bridging {self.source_width}x{self.source_height} @ {self.source_fps:.2f} fps "
                f"to {self.output_width}x{self.output_height} @ {self.publish_fps:.2f} fps "
                f"(stride {self.frame_stride}) from {self.video_path}"
            )
        )

        self.timer = self.create_timer(1.0 / self.publish_fps, self._publish_next_frame)

    def _build_camera_info_template(self) -> CameraInfo:
        if self.camera_info_mode == "approx":
            return build_approx_camera_info(
                width=self.output_width,
                height=self.output_height,
                horizontal_fov_deg=self.horizontal_fov_deg,
                frame_id=self.frame_id,
            )
        if self.camera_info_mode == "yaml":
            if not self.camera_info_yaml:
                raise RuntimeError("camera_info_mode=yaml requires camera_info_yaml to be set.")
            return load_camera_info_from_yaml(
                yaml_path=self.camera_info_yaml,
                output_width=self.output_width,
                output_height=self.output_height,
                frame_id=self.frame_id,
            )
        raise RuntimeError(
            f"Unsupported camera_info_mode '{self.camera_info_mode}'. Use 'approx' or 'yaml'."
        )

    def _read_publish_frame(self) -> tuple[int, object] | tuple[None, None]:
        while self.current_source_frame_index < self.next_source_frame_index:
            if not self.cap.grab():
                return None, None
            self.current_source_frame_index += 1

        ok, frame = self.cap.read()
        if not ok:
            return None, None

        frame_index = self.current_source_frame_index
        self.current_source_frame_index += 1
        self.next_source_frame_index += self.frame_stride
        return frame_index, frame

    def _publish_next_frame(self) -> None:
        if self.replay_start_time is None:
            self.replay_start_time = self.get_clock().now()

        frame_index, frame = self._read_publish_frame()
        if frame is None:
            self.timer.cancel()
            if self.frames_published == 0:
                raise RuntimeError(
                    f"Video decoder opened {self.video_path} but could not read the first frame."
                )
            self.get_logger().info("Video replay finished.")
            return

        if frame.shape[1] != self.output_width or frame.shape[0] != self.output_height:
            frame = cv2.resize(frame, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)

        stamp = (self.replay_start_time + Duration(seconds=frame_index / self.source_fps)).to_msg()

        camera_info_msg = copy.deepcopy(self.camera_info_template)
        camera_info_msg.header.stamp = stamp
        camera_info_msg.header.frame_id = self.frame_id

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id

        self.camera_info_pub.publish(camera_info_msg)
        self.image_pub.publish(image_msg)
        self.frames_published += 1


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VideoBridge()
    try:
        rclpy.spin(node)
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
