# Chapter 6: Capstone Project — Building an Autonomous Humanoid Robot

## Introduction to the Capstone Project

This capstone project integrates all concepts covered in previous chapters to build a fully autonomous humanoid robot capable of understanding natural language commands, navigating environments, recognizing and manipulating objects, and executing complex tasks. The project demonstrates the complete pipeline from voice input to physical action execution.

The autonomous humanoid robot will feature:
- Voice command recognition and interpretation
- Autonomous navigation and path planning
- Real-time obstacle detection and avoidance
- Object recognition and manipulation
- Multi-modal integration of vision, language, and action

## System Architecture Overview

The complete system architecture combines multiple components working in harmony:

```
Voice Command → Speech Recognition → LLM Interpretation → Task Planning
      ↓
Vision Processing → Object Recognition → Environment Mapping
      ↓
Navigation Planning → Path Execution → Manipulation Control
      ↓
Task Completion → Feedback Generation
```

### Core Components Integration

The system integrates:

- **ROS 2 Humble**: Communication middleware
- **Isaac Sim**: Simulation environment
- **OpenAI Whisper**: Speech recognition
- **GPT Model**: Natural language understanding
- **Nav2**: Navigation and path planning
- **Computer Vision**: Object detection and recognition
- **Control Systems**: Joint control and manipulation

## Voice Command Input and Processing

### Speech Recognition Pipeline

The voice command processing pipeline handles natural language input:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import whisper
import openai
import threading
import pyaudio
import wave
from queue import Queue

class VoiceCommandProcessor(Node):
    def __init__(self):
        super().__init__('voice_command_processor')
        
        # Initialize Whisper model
        self.whisper_model = whisper.load_model("base.en")
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.get_parameter('openai_api_key').value)
        
        # Publishers and subscribers
        self.command_pub = self.create_publisher(String, '/humanoid/command_parsed', 10)
        self.response_pub = self.create_publisher(String, '/humanoid/response', 10)
        
        # Audio recording setup
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        
        # Audio processing
        self.audio_queue = Queue()
        self.recording = False
        self.audio_thread = threading.Thread(target=self.continuous_recording)
        self.audio_thread.daemon = True
        
        # Start audio recording
        self.start_voice_recognition()
    
    def start_voice_recognition(self):
        """Start continuous voice recognition"""
        self.recording = True
        self.audio_thread.start()
    
    def continuous_recording(self):
        """Continuously record and process audio"""
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        frames = []
        
        while self.recording:
            data = stream.read(self.chunk)
            frames.append(data)
            
            # Process every 3 seconds of audio
            if len(frames) * self.chunk // self.rate >= 3:
                self.process_audio_chunk(frames)
                frames = []  # Reset frames
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def process_audio_chunk(self, frames):
        """Process audio chunk with Whisper"""
        try:
            # Save temporary audio file
            temp_filename = f"/tmp/temp_audio_{self.get_clock().now().nanoseconds}.wav"
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_filename)
            text = result["text"].strip()
            
            if len(text) > 5:  # Minimum meaningful command length
                self.get_logger().info(f'Voice command: {text}')
                self.process_natural_language_command(text)
                
        except Exception as e:
            self.get_logger().error(f'Audio processing error: {e}')
    
    def process_natural_language_command(self, command):
        """Process natural language command with LLM"""
        try:
            # Get robot context
            robot_context = self.get_robot_context()
            
            # Prepare LLM prompt
            prompt = f"""
            Human Command: {command}
            
            Robot Context:
            - Location: {robot_context['location']}
            - Orientation: {robot_context['orientation']}
            - Battery Level: {robot_context['battery']}
            - Available Actions: {robot_context['available_actions']}
            
            Convert this command into a structured action plan that the robot can execute.
            The plan should include:
            1. Specific actions to take
            2. Target locations or objects
            3. Sequence of operations
            4. Safety considerations
            
            Respond in JSON format:
            {{
                "action_plan": [
                    {{
                        "action": "action_type",
                        "parameters": {{}},
                        "description": "human-readable description"
                    }}
                ],
                "confidence": 0.0-1.0,
                "understanding": "brief explanation of command interpretation"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a command interpreter for a humanoid robot. Convert natural language to structured action plans."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            plan_data = response.choices[0].message.content
            import json
            action_plan = json.loads(plan_data)
            
            # Validate and publish action plan
            if action_plan.get("confidence", 0) > 0.7:
                cmd_msg = String()
                cmd_msg.data = json.dumps(action_plan)
                self.command_pub.publish(cmd_msg)
                
                # Provide feedback to user
                feedback_msg = String()
                feedback_msg.data = f"I understand you want me to: {action_plan.get('understanding', 'perform a task')}"
                self.response_pub.publish(feedback_msg)
            else:
                feedback_msg = String()
                feedback_msg.data = "I'm not confident I understood your command correctly. Could you please repeat or rephrase?"
                self.response_pub.publish(feedback_msg)
                
        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
            feedback_msg = String()
            feedback_msg.data = "I encountered an error processing your command. Please try again."
            self.response_pub.publish(feedback_msg)
```

## Planning and Navigation with ROS 2 and Nav2

### Navigation System Integration

The navigation system handles path planning and obstacle avoidance:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import json

class NavigationSystem(Node):
    def __init__(self):
        super().__init__('navigation_system')
        
        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Publishers and subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.command_sub = self.create_subscription(String, '/humanoid/command_parsed', self.command_callback, 10)
        
        # Obstacle detection
        self.obstacle_threshold = 0.5  # meters
        self.obstacles_detected = False
        
        # Current robot state
        self.current_pose = None
        self.goal_pose = None
    
    def command_callback(self, msg):
        """Process parsed command and execute navigation"""
        try:
            command_data = json.loads(msg.data)
            action_plan = command_data.get('action_plan', [])
            
            for action in action_plan:
                if action['action'] == 'navigate':
                    self.navigate_to_location(action['parameters'])
                elif action['action'] == 'find_object':
                    self.search_for_object(action['parameters'])
                elif action['action'] == 'manipulate':
                    self.execute_manipulation(action['parameters'])
        
        except Exception as e:
            self.get_logger().error(f'Command processing error: {e}')
    
    def navigate_to_location(self, params):
        """Navigate to specified location"""
        goal_msg = NavigateToPose.Goal()
        
        # Create goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = params.get('x', 0.0)
        goal_pose.pose.position.y = params.get('y', 0.0)
        goal_pose.pose.position.z = params.get('z', 0.0)
        
        # Set orientation (default to facing forward)
        goal_pose.pose.orientation.w = 1.0
        
        goal_msg.pose = goal_pose
        
        # Send navigation goal
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_result_callback)
    
    def navigation_result_callback(self, future):
        """Handle navigation result"""
        try:
            goal_result = future.result()
            if goal_result.status == 2:  # SUCCEEDED
                self.get_logger().info('Navigation completed successfully')
                # Continue with next action in plan
            else:
                self.get_logger().error('Navigation failed')
        except Exception as e:
            self.get_logger().error(f'Navigation result error: {e}')
    
    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Check for obstacles in front of robot
        front_range = msg.ranges[len(msg.ranges)//2]  # Front reading
        
        if front_range < self.obstacle_threshold:
            self.obstacles_detected = True
            self.get_logger().warn('Obstacle detected in front of robot')
        else:
            self.obstacles_detected = False
```

## Obstacle Detection and Avoidance

### Real-time Obstacle Processing

The obstacle detection system ensures safe navigation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray
import numpy as np
from scipy.spatial import distance

class ObstacleAvoidanceSystem(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_system')
        
        # Publishers and subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacles_markers', 10)
        
        # Parameters
        self.safe_distance = 0.6  # meters
        self.avoidance_distance = 0.4  # meters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        
        # Obstacle detection
        self.obstacles = []
        self.closest_obstacle = None
    
    def scan_callback(self, scan_msg):
        """Process laser scan data for obstacle detection"""
        # Convert laser scan to Cartesian coordinates
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        valid_ranges = np.array(scan_msg.ranges)
        
        # Filter out invalid ranges
        valid_mask = (valid_ranges > scan_msg.range_min) & (valid_ranges < scan_msg.range_max)
        valid_angles = angles[valid_mask]
        valid_distances = valid_ranges[valid_mask]
        
        # Calculate Cartesian coordinates
        x_coords = valid_distances * np.cos(valid_angles)
        y_coords = valid_distances * np.sin(valid_angles)
        
        # Detect obstacles within safe distance
        obstacle_mask = valid_distances < self.safe_distance
        obstacle_x = x_coords[obstacle_mask]
        obstacle_y = y_coords[obstacle_mask]
        
        # Store obstacle information
        self.obstacles = list(zip(obstacle_x, obstacle_y))
        
        # Find closest obstacle in front of robot
        front_mask = (valid_angles > -0.5) & (valid_angles < 0.5)  # Front 90 degrees
        front_distances = valid_distances[front_mask]
        
        if len(front_distances) > 0:
            min_distance_idx = np.argmin(front_distances)
            self.closest_obstacle = front_distances[min_distance_idx]
        else:
            self.closest_obstacle = float('inf')
        
        # Execute obstacle avoidance
        self.execute_avoidance_behavior()
    
    def execute_avoidance_behavior(self):
        """Execute obstacle avoidance based on detected obstacles"""
        cmd_vel = Twist()
        
        if self.closest_obstacle is not None and self.closest_obstacle < self.avoidance_distance:
            # Stop forward motion
            cmd_vel.linear.x = 0.0
            
            # Determine turn direction based on obstacle distribution
            left_obstacles = [obs for obs in self.obstacles if obs[1] > 0]  # Right side
            right_obstacles = [obs for obs in self.obstacles if obs[1] < 0]  # Left side
            
            if len(left_obstacles) < len(right_obstacles):
                # Turn left (positive angular velocity)
                cmd_vel.angular.z = self.angular_speed
            else:
                # Turn right (negative angular velocity)
                cmd_vel.angular.z = -self.angular_speed
        else:
            # Safe to move forward
            cmd_vel.linear.x = self.linear_speed
            cmd_vel.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd_vel)
```

## Object Recognition and Manipulation

### Computer Vision Integration

Object recognition and manipulation system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np

class ObjectRecognitionSystem(Node):
    def __init__(self):
        super().__init__('object_recognition_system')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Load object detection model (YOLOv5 or similar)
        self.detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detection_model.eval()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Detected objects
        self.detected_objects = []
    
    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def image_callback(self, image_msg):
        """Process camera image for object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # Run object detection
            results = self.detection_model(cv_image)
            detections = results.pandas().xyxy[0].to_dict()
            
            # Process detections
            self.detected_objects = []
            for i in range(len(detections['name'])):
                obj = {
                    'class': detections['name'][i],
                    'confidence': float(detections['confidence'][i]),
                    'bbox': [float(detections['xmin'][i]), float(detections['ymin'][i]), 
                            float(detections['xmax'][i]), float(detections['ymax'][i])],
                    'center_pixel': ((detections['xmin'][i] + detections['xmax'][i]) / 2,
                                    (detections['ymin'][i] + detections['ymax'][i]) / 2)
                }
                
                # Calculate 3D position if camera parameters are available
                if self.camera_matrix is not None:
                    obj['position_3d'] = self.pixel_to_3d_position(
                        obj['center_pixel'], obj['bbox'][3] - obj['bbox'][1]  # Use height for depth estimation
                    )
                
                self.detected_objects.append(obj)
            
            # Log detected objects
            for obj in self.detected_objects:
                if obj['confidence'] > 0.5:  # Confidence threshold
                    self.get_logger().info(f'Detected {obj["class"]} with confidence {obj["confidence"]:.2f}')
        
        except Exception as e:
            self.get_logger().error(f'Object detection error: {e}')
    
    def pixel_to_3d_position(self, pixel, object_height_px):
        """Convert pixel coordinates to 3D world coordinates"""
        if self.camera_matrix is None:
            return None
        
        # This is a simplified conversion - in practice, you'd need depth information
        # or stereo vision for accurate 3D positioning
        u, v = pixel
        
        # Convert to normalized coordinates
        x_norm = (u - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_norm = (v - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        # For now, return normalized coordinates (depth would need to be estimated)
        return [x_norm, y_norm, 1.0]  # Placeholder for actual depth
    
    def find_object_by_class(self, obj_class, min_confidence=0.5):
        """Find object of specific class"""
        for obj in self.detected_objects:
            if (obj['class'] == obj_class and 
                obj['confidence'] >= min_confidence):
                return obj
        return None
    
    def get_object_position(self, obj_class):
        """Get 3D position of object if available"""
        obj = self.find_object_by_class(obj_class)
        if obj and 'position_3d' in obj:
            return obj['position_3d']
        return None
```

## Multi-Modal Integration: Vision, Language, and Action

### Complete Integration Pipeline

The main integration node that combines all systems:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import json
import time

class HumanoidIntegrationNode(Node):
    def __init__(self):
        super().__init__('humanoid_integration')
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/humanoid/status', 10)
        
        # Initialize subsystems
        self.voice_processor = VoiceCommandProcessor()
        self.navigation_system = NavigationSystem()
        self.obstacle_avoidance = ObstacleAvoidanceSystem()
        self.object_recognition = ObjectRecognitionSystem()
        
        # Task execution state
        self.current_task = None
        self.task_queue = []
        
        # Timer for task execution
        self.task_timer = self.create_timer(0.1, self.execute_task)
    
    def execute_task(self):
        """Execute tasks from the queue"""
        if self.task_queue and not self.current_task:
            self.current_task = self.task_queue.pop(0)
            self.execute_single_task(self.current_task)
    
    def execute_single_task(self, task):
        """Execute a single task based on its type"""
        task_type = task.get('action', '')
        
        if task_type == 'navigate':
            self.execute_navigation_task(task)
        elif task_type == 'grasp':
            self.execute_grasp_task(task)
        elif task_type == 'search':
            self.execute_search_task(task)
        elif task_type == 'speak':
            self.execute_speak_task(task)
    
    def execute_navigation_task(self, task):
        """Execute navigation task"""
        self.get_logger().info(f'Navigating to: {task.get("parameters", {})}')
        self.navigation_system.navigate_to_location(task.get('parameters', {}))
        
        # Mark task as complete after delay (in real system, this would be event-driven)
        time.sleep(2)
        self.current_task = None
        self.publish_status('Navigation task completed')
    
    def execute_grasp_task(self, task):
        """Execute object grasping task"""
        obj_class = task.get('parameters', {}).get('object_class', '')
        
        # Find object using vision system
        obj_position = self.object_recognition.get_object_position(obj_class)
        
        if obj_position:
            self.get_logger().info(f'Grasping {obj_class} at position {obj_position}')
            # Execute grasp action (implementation would control actual robot)
            self.publish_status(f'Grasped {obj_class}')
        else:
            self.get_logger().warn(f'Could not find {obj_class} to grasp')
            self.publish_status(f'Failed to find {obj_class}')
        
        self.current_task = None
    
    def execute_search_task(self, task):
        """Execute object search task"""
        obj_class = task.get('parameters', {}).get('object_class', '')
        
        # Use vision system to search for object
        obj = self.object_recognition.find_object_by_class(obj_class)
        
        if obj:
            self.get_logger().info(f'Found {obj_class} with confidence {obj["confidence"]:.2f}')
            self.publish_status(f'Found {obj_class}')
        else:
            self.get_logger().info(f'Did not find {obj_class}')
            self.publish_status(f'Did not find {obj_class}')
        
        self.current_task = None
    
    def execute_speak_task(self, task):
        """Execute speech task"""
        text = task.get('parameters', {}).get('text', '')
        self.get_logger().info(f'Speaking: {text}')
        # In real system, this would interface with text-to-speech
        self.publish_status(f'Spoken: {text}')
        self.current_task = None
    
    def publish_status(self, status):
        """Publish status message"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
```

## End-to-End Workflow Implementation

### Complete Example: Fetch and Deliver Task

Here's a complete example of the end-to-end workflow:

```python
def complete_fetch_task_example():
    """
    Example workflow: "Robot, please bring me the red cup from the table"
    
    1. Voice recognition with Whisper
    2. LLM interpretation
    3. Object recognition
    4. Navigation planning
    5. Grasping execution
    6. Delivery
    """
    
    # Step 1: Voice command received and processed by Whisper
    voice_command = "bring me the red cup from the table"
    
    # Step 2: LLM converts to structured plan
    action_plan = {
        "action_plan": [
            {
                "action": "search",
                "parameters": {"object_class": "cup", "color": "red"},
                "description": "Search for red cup"
            },
            {
                "action": "navigate",
                "parameters": {"x": 2.0, "y": 1.5, "z": 0.0},
                "description": "Navigate to table location"
            },
            {
                "action": "grasp",
                "parameters": {"object_class": "cup"},
                "description": "Grasp the red cup"
            },
            {
                "action": "navigate",
                "parameters": {"x": 0.0, "y": 0.0, "z": 0.0},
                "description": "Return to user"
            },
            {
                "action": "speak",
                "parameters": {"text": "I have brought you the red cup"},
                "description": "Inform user of completion"
            }
        ],
        "confidence": 0.9,
        "understanding": "Fetch red cup from table and deliver to user"
    }
    
    # Step 3: Task execution through integration node
    integration_node = HumanoidIntegrationNode()
    
    # Queue tasks for execution
    for task in action_plan["action_plan"]:
        integration_node.task_queue.append(task)
    
    # System executes tasks in sequence
    # Each task triggers appropriate subsystems:
    # - Vision system for object recognition
    # - Navigation system for movement
    # - Control system for manipulation
    # - Speech system for communication
    
    print("End-to-end task execution completed successfully!")
```

## Best Practices for Simulation-to-Real Deployment

### Key Considerations

When transitioning from simulation to real hardware:

#### 1. Domain Randomization in Simulation
- Train perception systems with varied lighting, textures, and environments
- Add noise to sensor data to match real-world conditions
- Vary physical parameters to account for real-world uncertainties

#### 2. Safety First Approach
- Implement multiple safety layers and emergency stops
- Test extensively in simulation before real-world deployment
- Use velocity and force limits during initial real-world testing

#### 3. Gradual Complexity Increase
- Start with simple, well-defined tasks
- Gradually increase task complexity as system proves reliable
- Monitor system performance and adjust parameters accordingly

#### 4. Robust Error Handling
- Implement fallback behaviors for failed actions
- Design systems that can recover from unexpected situations
- Log all errors for analysis and improvement

## Conclusion

This capstone project demonstrates the integration of all major components required for an autonomous humanoid robot: voice recognition, natural language understanding, navigation, obstacle avoidance, object recognition, and manipulation. The system architecture provides a foundation for building more sophisticated autonomous robots that can interact naturally with humans and perform complex tasks in real-world environments.

The key to success lies in the seamless integration of these components and the careful consideration of safety, reliability, and user experience. As robotics technology continues to advance, these integrated systems will become increasingly capable and ubiquitous, transforming how humans interact with their environment through intelligent robotic assistants.