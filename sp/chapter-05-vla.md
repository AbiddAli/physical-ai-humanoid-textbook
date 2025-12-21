# Chapter 5: Vision-Language-Action (VLA) — Integrating LLMs with Humanoid Robots

## Introduction to Vision-Language-Action Systems

Vision-Language-Action (VLA) systems represent a paradigm shift in robotics, where humanoid robots can understand natural language commands, perceive their environment visually, and execute complex tasks through coordinated actions. This integration of multimodal AI enables robots to interact with humans more naturally and perform complex tasks without explicit programming for each scenario.

VLA systems combine three critical components:
- **Vision**: Real-time perception of the environment
- **Language**: Understanding and generating natural language
- **Action**: Executing physical tasks based on perception and language understanding

For humanoid robots, VLA systems enable unprecedented levels of autonomy and human-robot interaction, moving beyond pre-programmed behaviors to truly intelligent, adaptive systems.

## Integrating LLMs with Humanoid Robots

### Large Language Models in Robotics Context

Large Language Models (LLMs) like GPT, Claude, and open-source alternatives provide powerful natural language understanding capabilities. When integrated with humanoid robots, LLMs can:

- Interpret complex, natural language commands
- Generate contextual responses to human queries
- Plan multi-step tasks based on high-level goals
- Learn from interactions to improve performance

### Architecture for LLM Integration

The integration architecture typically involves several layers:

```python
import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Duration

class LLMHumanoidInterface(Node):
    def __init__(self):
        super().__init__('llm_humanoid_interface')
        
        # Initialize LLM client
        self.llm_client = openai.OpenAI(api_key=self.get_parameter('openai_api_key').value)
        
        # ROS 2 publishers and subscribers
        self.command_sub = self.create_subscription(
            String, '/humanoid/voice_command', self.voice_command_callback, 10
        )
        
        self.image_sub = self.create_subscription(
            Image, '/humanoid/camera/image_raw', self.image_callback, 10
        )
        
        self.response_pub = self.create_publisher(String, '/humanoid/response', 10)
        
        # Action execution interface
        self.action_client = ActionClient(self, ExecuteTask, '/execute_task')
        
        # Memory system for context
        self.conversation_history = []
        self.max_history_length = 10
    
    def voice_command_callback(self, msg):
        """Process voice command through LLM"""
        try:
            # Get current robot state and environment context
            context = self.get_robot_context()
            
            # Prepare prompt for LLM
            prompt = self.build_prompt(msg.data, context)
            
            # Generate response using LLM
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    *self.conversation_history,
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Extract action plan from response
            action_plan = self.parse_action_plan(response.choices[0].message.content)
            
            # Execute action plan
            self.execute_action_plan(action_plan)
            
            # Update conversation history
            self.update_conversation_history(msg.data, response.choices[0].message.content)
            
        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
    
    def get_system_prompt(self):
        """Define system behavior for LLM"""
        return """
        You are an intelligent assistant for a humanoid robot. Your role is to:
        1. Understand natural language commands
        2. Generate step-by-step action plans for the robot
        3. Consider the robot's capabilities and environment
        4. Respond in a structured format that the robot can parse
        
        Robot capabilities:
        - Walk to locations
        - Grasp objects
        - Manipulate objects
        - Navigate obstacles
        - Speak responses
        
        Format your response as:
        ACTION_PLAN:
        1. [Action description]
        2. [Action description]
        ...
        
        RESPONSE: [Natural language response to human]
        """
    
    def parse_action_plan(self, llm_response):
        """Parse structured action plan from LLM response"""
        import re
        
        # Extract action plan
        action_match = re.search(r'ACTION_PLAN:\n((?:\d+\.\s[^\n]+\n?)*)', llm_response)
        if action_match:
            actions = []
            for line in action_match.group(1).strip().split('\n'):
                if line.strip():
                    # Extract action number and description
                    action_desc = re.sub(r'^\d+\.\s*', '', line.strip())
                    actions.append(action_desc)
            return actions
        
        return []
    
    def execute_action_plan(self, action_plan):
        """Execute parsed action plan"""
        for action in action_plan:
            if "walk to" in action.lower():
                # Parse location and navigate
                location = self.extract_location(action)
                self.navigate_to_location(location)
            elif "grasp" in action.lower() or "pick up" in action.lower():
                # Parse object and grasp
                obj = self.extract_object(action)
                self.grasp_object(obj)
            elif "place" in action.lower() or "put" in action.lower():
                # Parse placement location
                location = self.extract_location(action)
                self.place_object(location)
            else:
                self.get_logger().info(f'Unknown action: {action}')
    
    def update_conversation_history(self, user_input, response):
        """Maintain conversation context"""
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
```

### Cognitive Planning with LLMs

LLMs excel at cognitive planning by breaking down complex tasks into executable steps:

```python
class CognitivePlanner:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.task_library = self.load_task_library()
    
    def plan_complex_task(self, natural_language_goal, robot_state, environment_state):
        """Generate executable plan from natural language goal"""
        
        # Create detailed context
        context = {
            "robot_capabilities": self.get_robot_capabilities(),
            "environment_objects": environment_state["objects"],
            "robot_location": robot_state["location"],
            "robot_orientation": robot_state["orientation"],
            "available_tools": self.get_available_tools()
        }
        
        # Build comprehensive prompt
        prompt = f"""
        Task Goal: {natural_language_goal}
        
        Robot Capabilities: {context['robot_capabilities']}
        Environment Objects: {context['environment_objects']}
        Robot Location: {context['robot_location']}
        Available Tools: {context['available_tools']}
        
        Generate a detailed action plan to achieve the goal. Each action should be:
        1. Specific and executable
        2. Sequentially ordered
        3. Account for robot limitations
        4. Include safety considerations
        
        Format as JSON:
        {{
            "actions": [
                {{
                    "id": 1,
                    "action": "description",
                    "parameters": {{}},
                    "preconditions": [],
                    "effects": []
                }}
            ],
            "estimated_duration": "time estimate",
            "safety_risks": ["potential risks"]
        }}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a cognitive planning assistant for robotics. Generate detailed, executable action plans."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            plan_data = response.choices[0].message.content
            import json
            return json.loads(plan_data)
            
        except Exception as e:
            self.get_logger().error(f'Planning error: {e}')
            return None
```

## Voice-to-Action: OpenAI Whisper Integration

### Speech Recognition with Whisper

OpenAI Whisper provides robust speech-to-text capabilities for humanoid robots:

```python
import whisper
import torch
import pyaudio
import wave
import threading
from queue import Queue

class VoiceToActionInterface:
    def __init__(self, robot_node):
        self.robot_node = robot_node
        self.whisper_model = whisper.load_model("base.en")  # Use "base" for real-time performance
        
        # Audio recording setup
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5
        
        # Audio queue for processing
        self.audio_queue = Queue()
        
        # Start audio recording thread
        self.recording = False
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.daemon = True
    
    def start_listening(self):
        """Start continuous voice command listening"""
        self.recording = True
        self.audio_thread.start()
    
    def record_audio(self):
        """Continuously record audio and process voice commands"""
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
            
            # Process audio every 5 seconds of recording
            if len(frames) * self.chunk // self.rate >= 5:
                # Save to temporary WAV file
                temp_filename = "temp_audio.wav"
                wf = wave.open(temp_filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # Process with Whisper
                self.process_voice_command(temp_filename)
                
                # Clear frames
                frames = []
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def process_voice_command(self, audio_file):
        """Process audio file with Whisper and send to LLM"""
        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(audio_file)
            text = result["text"].strip()
            
            # Only process if text is meaningful
            if len(text) > 3:  # Minimum length check
                self.robot_node.get_logger().info(f'Voice command: {text}')
                
                # Send to LLM for processing
                cmd_msg = String()
                cmd_msg.data = text
                self.robot_node.voice_command_callback(cmd_msg)
                
        except Exception as e:
            self.robot_node.get_logger().error(f'Whisper processing error: {e}')
    
    def stop_listening(self):
        """Stop audio recording"""
        self.recording = False
        if self.audio_thread.is_alive():
            self.audio_thread.join()
```

### Voice Command Processing Pipeline

The complete voice processing pipeline includes:

1. **Audio Capture**: Continuous recording with noise filtering
2. **Speech Recognition**: Whisper transcription
3. **Natural Language Processing**: LLM interpretation
4. **Action Execution**: ROS 2 command execution

## Multi-Modal Perception Integration

### Combining Vision, Language, and Action

Multi-modal perception systems combine inputs from multiple sensors to create a comprehensive understanding:

```python
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MultiModalPerception:
    def __init__(self, robot_node):
        self.robot_node = robot_node
        self.cv_bridge = CvBridge()
        
        # Vision processing
        self.object_detector = self.initialize_object_detector()
        self.pose_estimator = self.initialize_pose_estimator()
        
        # State tracking
        self.current_scene = {}
        self.object_memory = {}
    
    def initialize_object_detector(self):
        """Initialize object detection model"""
        # Using YOLO or similar for real-time object detection
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    def process_visual_input(self, image_msg):
        """Process camera image for object detection and scene understanding"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # Run object detection
            results = self.object_detector(cv_image)
            detections = results.pandas().xyxy[0].to_dict()
            
            # Extract relevant objects
            objects = []
            for i in range(len(detections['name'])):
                obj = {
                    'class': detections['name'][i],
                    'confidence': detections['confidence'][i],
                    'bbox': [detections['xmin'][i], detections['ymin'][i], 
                            detections['xmax'][i], detections['ymax'][i]],
                    'center': [(detections['xmin'][i] + detections['xmax'][i]) / 2,
                              (detections['ymin'][i] + detections['ymax'][i]) / 2]
                }
                objects.append(obj)
            
            # Update scene understanding
            self.current_scene = {
                'timestamp': image_msg.header.stamp,
                'objects': objects,
                'image_shape': cv_image.shape
            }
            
            return self.current_scene
            
        except Exception as e:
            self.robot_node.get_logger().error(f'Vision processing error: {e}')
            return None
    
    def integrate_multimodal_input(self, text_command, visual_context, robot_state):
        """Integrate text, vision, and robot state for action planning"""
        
        # Create comprehensive context
        context = {
            'language_input': text_command,
            'visual_context': visual_context,
            'robot_state': robot_state,
            'environment_map': self.get_environment_map(),
            'object_memory': self.object_memory
        }
        
        # Use LLM to interpret and plan
        interpretation = self.interpret_multimodal_context(context)
        
        return interpretation
    
    def interpret_multimodal_context(self, context):
        """Interpret combined modalities using LLM"""
        
        prompt = f"""
        Language Command: {context['language_input']}
        Visual Objects: {[obj['class'] for obj in context['visual_context']['objects']]}
        Robot Location: {context['robot_state']['location']}
        Robot Capabilities: {context['robot_state']['capabilities']}
        
        Based on the language command and visual context, determine:
        1. What the user wants the robot to do
        2. Which objects are relevant
        3. What actions are needed
        4. Any potential obstacles or considerations
        
        Provide response in structured format.
        """
        
        # Process with LLM (implementation similar to previous examples)
        # Return structured interpretation
        pass
```

## Practical Examples of Humanoid Task Execution

### Example 1: Fetch and Deliver Task

A practical example of a complex task execution:

```python
def execute_fetch_and_deliver_task(robot, command):
    """
    Example: "Robot, please bring me the red cup from the kitchen table"
    """
    
    # Step 1: Parse command with LLM
    action_plan = [
        "Navigate to kitchen",
        "Locate red cup on table", 
        "Approach table",
        "Grasp red cup",
        "Navigate to user",
        "Extend arm to present cup"
    ]
    
    # Step 2: Execute navigation
    kitchen_location = robot.get_location("kitchen")
    robot.navigate_to(kitchen_location)
    
    # Step 3: Object recognition
    objects = robot.perceive_environment()
    red_cup = find_object_by_color_and_type(objects, "red", "cup")
    
    # Step 4: Grasping
    robot.approach_object(red_cup)
    robot.grasp_object(red_cup)
    
    # Step 5: Return to user
    user_location = robot.get_user_location()
    robot.navigate_to(user_location)
    
    # Step 6: Present object
    robot.present_object(red_cup)

def find_object_by_color_and_type(objects, color, obj_type):
    """Find object matching color and type criteria"""
    for obj in objects:
        if obj['type'] == obj_type and obj['color'] == color:
            return obj
    return None
```

### Example 2: Complex Multi-Step Task

```python
def execute_cleanup_task(robot, command):
    """
    Example: "Clean up the living room by putting books on the shelf and cups in the kitchen"
    """
    
    # Parse complex command
    tasks = [
        {
            "action": "find_objects",
            "criteria": {"type": "book"},
            "destination": "bookshelf"
        },
        {
            "action": "find_objects", 
            "criteria": {"type": "cup"},
            "destination": "kitchen_counter"
        }
    ]
    
    for task in tasks:
        # Find all objects matching criteria
        objects = robot.find_objects_by_criteria(task["criteria"])
        
        for obj in objects:
            # Navigate to object
            robot.navigate_to(obj["location"])
            
            # Grasp object
            robot.grasp_object(obj)
            
            # Navigate to destination
            destination = robot.get_location(task["destination"])
            robot.navigate_to(destination)
            
            # Place object
            robot.place_object(obj, destination)
```

## Best Practices and Considerations

### Safety and Validation

- **Command Validation**: Verify LLM-generated commands are safe before execution
- **Emergency Override**: Implement manual stop capabilities
- **Context Awareness**: Ensure the robot understands its environment constraints

### Performance Optimization

- **Caching**: Store frequently accessed information to reduce LLM calls
- **Parallel Processing**: Process perception and language tasks concurrently
- **Edge Computing**: Use local models when possible to reduce latency

### Privacy and Security

- **Data Protection**: Securely handle personal information in conversations
- **Access Control**: Implement authentication for sensitive commands
- **Data Minimization**: Collect only necessary data for task execution

## Conclusion

Vision-Language-Action systems represent the future of human-robot interaction, enabling humanoid robots to understand and execute complex, natural language commands. The integration of LLMs, speech recognition, and multi-modal perception creates intelligent systems that can adapt to novel situations and interact naturally with humans.

As we continue to develop these systems, the focus must remain on safety, reliability, and user experience. The combination of advanced AI with robust robotic platforms will enable humanoid robots to become truly useful assistants in homes, workplaces, and public spaces.

The foundation laid in this chapter—understanding how to integrate multimodal AI with robotic systems—will be essential as we move toward the capstone project that combines all the concepts explored throughout this textbook.