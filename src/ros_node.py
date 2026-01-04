import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import cv2
import numpy as np

# Import our core detectors
from src.road_damage.detector import RoadDamageDetector
from src.lane_detection.classical_lane import ClassicalLaneDetector
from src.utils.visualization import draw_dashboard, draw_lanes, draw_damage

class ADASNode(Node):
    def __init__(self):
        super().__init__('adas_perception_node')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        self.get_logger().info(f"Initializing ADAS Node with model: {model_path}")
        
        # Initialize Detectors
        try:
            self.damage_detector = RoadDamageDetector(model_path=model_path)
            self.lane_detector = ClassicalLaneDetector()
        except Exception as e:
            self.get_logger().error(f"Failed to load detectors: {e}")
            raise e

        # ROS Comm
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.listener_callback,
            10
        )
        self.publisher_vis = self.create_publisher(Image, '/adas/visualization', 10)
        self.publisher_alert = self.create_publisher(String, '/adas/alerts', 10)
        
        self.cv_bridge = CvBridge()
        self.get_logger().info("ADAS Node Ready.")

    def listener_callback(self, msg):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Inference
        # 1. Road Damage
        boxes, scores, class_ids = self.damage_detector.detect(frame)
        
        # 2. Lane Detection
        left_line, right_line = self.lane_detector.process(frame)
        
        # Publish Alerts if damage found
        if len(boxes) > 0:
            alert_msg = String()
            alert_msg.data = f"Detected {len(boxes)} road hazards!"
            self.publisher_alert.publish(alert_msg)

        # Visualization for Rviz/Debug
        viz_frame = draw_lanes(frame.copy(), left_line, right_line)
        viz_frame = draw_damage(viz_frame, boxes, scores, class_ids, self.damage_detector.get_class_names())
        
        try:
            out_msg = self.cv_bridge.cv2_to_imgmsg(viz_frame, 'bgr8')
            self.publisher_vis.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Visualization Publish Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ADASNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
