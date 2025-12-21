# Chapter 2: ROS 2 Fundamentals

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is the next-generation middleware framework designed for robotics applications. Unlike its predecessor, ROS 2 addresses critical issues such as real-time performance, security, and deployment in production environments. For Physical AI systems, particularly humanoid robots, ROS 2 provides the essential communication infrastructure that enables seamless interaction between perception, planning, and control modules.

ROS 2 implements a client library architecture that supports multiple programming languages while maintaining consistent messaging semantics. The framework utilizes Data Distribution Service (DDS) as its underlying communication layer, ensuring reliable message delivery even in distributed systems with multiple robots or computational units.

## ROS 2 Architecture

### Client Libraries

ROS 2 supports multiple client libraries, with the most common being:

- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rclrs**: Rust client library (experimental)
- **rclc**: C client library for embedded systems

For Physical AI applications involving humanoid robots, rclpy is often preferred for rapid prototyping due to Python's ease of use, while rclcpp is chosen for performance-critical components like real-time controllers.

### DDS Implementations

ROS 2 abstracts the underlying DDS implementation, allowing users to switch between:

- **Fast DDS** (default in Humble): Developed by eProsima, offers excellent performance
- **Cyclone DDS**: Lightweight implementation from Eclipse
- **RTI Connext DDS**: Commercial solution with enterprise features

The choice of DDS implementation can affect latency and throughput, which are critical for real-time humanoid robot control.

## Core Concepts

### Nodes

A node is the fundamental unit of computation in ROS 2. Each node performs a specific function and communicates with other nodes through topics, services, or actions. In humanoid robotics, nodes typically handle:

- Sensor data processing
- Control algorithms
- State estimation
- Behavior planning

Here's a minimal ROS 2 node example in Python:

```python
import rclpy
from rclpy.node import Node

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        # Initialize publishers, subscribers, timers, etc.
        self.get_logger().info('Joint State Publisher initialized')

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics

Topics enable asynchronous, many-to-many communication between nodes. Publishers send messages to topics, and subscribers receive messages from topics. This decoupled communication pattern is ideal for continuous data streams like sensor readings or joint states.

For humanoid robots, common topics include:
- `/joint_states`: Current joint positions, velocities, and efforts
- `/tf` and `/tf_static`: Coordinate transforms between robot frames
- `/imu/data`: Inertial measurement unit readings
- `/camera/image_raw`: Camera image streams

Example topic publisher:

```python
from std_msgs.msg import Float64MultiArray
import rclpy
from rclpy.node import Node

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher = self.create_publisher(
            Float64MultiArray, 
            '/joint_commands', 
            10
        )
        
    def publish_joint_commands(self, commands):
        msg = Float64MultiArray()
        msg.data = commands
        self.publisher.publish(msg)
```

### Services

Services provide synchronous request-response communication. A service client sends a request and waits for a response from a service server. This pattern is useful for operations that require acknowledgment or return specific results.

Common service types in humanoid robotics:
- `/set_parameters`: Dynamically configure node parameters
- Custom services for robot calibration or diagnostic routines

Service server example:

```python
from example_interfaces.srv import SetBool
import rclpy
from rclpy.node import Node

class RobotEnableService(Node):
    def __init__(self):
        super().__init__('robot_enable_service')
        self.srv = self.create_service(
            SetBool, 
            'enable_robot', 
            self.enable_robot_callback
        )
        self.robot_enabled = False
    
    def enable_robot_callback(self, request, response):
        self.robot_enabled = request.data
        response.success = True
        response.message = f'Robot enabled: {self.robot_enabled}'
        return response
```

### Actions

Actions are designed for long-running tasks that require feedback and the ability to cancel. They consist of three parts:
- Goal: Request to start a task
- Feedback: Periodic updates during task execution
- Result: Final outcome when task completes

Actions are crucial for humanoid robot tasks like:
- Walking to a location
- Grasping objects
- Following trajectories

Action client example:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory

class TrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('trajectory_executor')
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            'follow_joint_trajectory'
        )
    
    def send_goal(self, trajectory_points):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory_points
        
        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg)
```

## Building ROS 2 Packages with Python

### Package Structure

A typical ROS 2 Python package follows this structure:

```
my_robot_package/
├── setup.py
├── package.xml
├── CMakeLists.txt
├── resource/my_robot_package
├── my_robot_package/
│   ├── __init__.py
│   ├── publisher_member_function.py
│   └── subscriber_member_function.py
└── test/
    └── test_copyright.py
```

### Creating a Package

Use the `ros2 pkg create` command to generate a new package:

```bash
ros2 pkg create --build-type ament_python my_humanoid_control
```

### setup.py Configuration

The `setup.py` file defines how Python modules are installed:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_humanoid_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Humanoid robot control package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_controller = my_humanoid_control.joint_controller:main',
            'state_estimator = my_humanoid_control.state_estimator:main',
        ],
    },
)
```

### Dependencies

Dependencies are declared in `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_humanoid_control</name>
  <version>0.0.0</version>
  <description>Humanoid robot control package</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>control_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Launch Files and Parameter Management

### Launch Files

Launch files allow you to start multiple nodes with specific configurations simultaneously. For humanoid robots, launch files typically initialize the complete robot stack:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'),
            
        Node(
            package='my_humanoid_control',
            executable='joint_controller',
            name='joint_controller',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),
            
        Node(
            package='my_humanoid_control',
            executable='state_estimator',
            name='state_estimator',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),
    ])
```

### Parameter Management

Parameters control node behavior without recompilation. For humanoid robots, parameters might include:

- Control gains
- Joint limits
- Safety thresholds
- Calibration values

Parameter declaration in nodes:

```python
class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        
        # Declare parameters with defaults
        self.declare_parameter('kp', 10.0)
        self.declare_parameter('ki', 0.1)
        self.declare_parameter('kd', 0.01)
        self.declare_parameter('max_velocity', 1.0)
        
        # Access parameters
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.max_velocity = self.get_parameter('max_velocity').value
```

Parameter files (YAML format):

```yaml
/**:
  ros__parameters:
    kp: 15.0
    ki: 0.2
    kd: 0.05
    max_velocity: 2.0
    joint_names: ["hip_joint", "knee_joint", "ankle_joint"]
```

## URDF for Humanoid Robots

Unified Robot Description Format (URDF) describes robot kinematics and dynamics. For humanoid robots, URDF becomes complex due to multiple limbs and degrees of freedom.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" 
               iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Hip joint and link -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_leg"/>
    <origin xyz="0 0.1 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
  
  <link name="left_leg">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" 
               iyy="0.005" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

</robot>
```

### Xacro for Complex Humanoid Models

Xacro (XML Macros) simplifies complex humanoid URDF definitions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="leg_length" value="0.4" />
  <xacro:property name="leg_radius" value="0.05" />
  
  <!-- Macro for leg definition -->
  <xacro:macro name="leg" params="side reflect">
    <joint name="${side}_hip_joint" type="revolute">
      <parent link="torso"/>
      <child link="${side}_upper_leg"/>
      <origin xyz="0 ${reflect*0.1} -0.1" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
    </joint>
    
    <link name="${side}_upper_leg">
      <visual>
        <geometry>
          <cylinder length="${leg_length}" radius="${leg_radius}"/>
        </geometry>
        <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${leg_length}" radius="${leg_radius}"/>
        </geometry>
        <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.005" ixy="0.0" ixz="0.0" 
                 iyy="0.005" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>
  
  <!-- Instantiate legs -->
  <xacro:leg side="left" reflect="1"/>
  <xacro:leg side="right" reflect="-1"/>
  
</robot>
```

## Best Practices for Physical AI

### Real-time Considerations

- Use dedicated threads for time-critical control loops
- Minimize message callback processing time
- Implement proper rate limiting for sensor data processing

### Safety Features

- Implement emergency stop mechanisms
- Set joint position and velocity limits
- Monitor control effort to detect anomalies

### Debugging Strategies

- Use RViz2 for visualization of robot state
- Log critical variables for offline analysis
- Implement parameter reconfiguration for tuning

## Conclusion

ROS 2 provides the essential infrastructure for developing Physical AI systems, particularly humanoid robots. Its flexible communication patterns, robust package management, and comprehensive tooling ecosystem make it the de facto standard for robotics development. Understanding nodes, topics, services, and actions is crucial for building modular, maintainable robot systems.

The combination of URDF for robot description and launch files for system initialization enables rapid prototyping and deployment of complex humanoid robot behaviors. As we proceed to subsequent chapters, we'll explore how these fundamental ROS 2 concepts integrate with advanced control strategies and perception systems to create truly intelligent physical agents.