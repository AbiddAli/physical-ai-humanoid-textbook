# Chapter 3: Digital Twin — Gazebo & Unity Simulation for Humanoid Robots

## Introduction to Digital Twins in Robotics

A digital twin in robotics is a virtual replica of a physical robot that mirrors its behavior, appearance, and interactions with the environment in real-time. For humanoid robots, digital twins serve as essential tools for development, testing, and validation before deploying to real hardware. This chapter explores two leading simulation environments—Gazebo and Unity—and their roles in creating accurate digital twins for humanoid robots.

Digital twins enable developers to:
- Test control algorithms without risk to physical hardware
- Validate sensor data processing pipelines
- Perform safety-critical experiments in controlled environments
- Accelerate development cycles through parallel simulation and real-world testing

## Gazebo Simulation Environment

Gazebo is the premier open-source physics simulator for robotics, widely adopted in the ROS ecosystem. Its realistic physics engine and tight integration with ROS 2 make it ideal for humanoid robot simulation.

### Physics Engine Capabilities

Gazebo utilizes the Open Dynamics Engine (ODE), Bullet Physics, or DART as its underlying physics engines. For humanoid robots, physics accuracy is crucial for:

- **Balance Control**: Accurate center of mass calculations for stable walking
- **Contact Dynamics**: Realistic foot-ground and hand-object interactions
- **Collision Detection**: Precise collision handling for safety
- **Friction Modeling**: Proper grip and slip simulation for manipulation tasks

### Environment Building in Gazebo

Creating realistic environments involves several components:

#### World Definition

World files define the simulation environment using SDF (Simulation Description Format):

```xml
<sdf version='1.7'>
  <world name='humanoid_world'>
    <!-- Include models from Fuel database -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Lighting -->
    <light name='sun' type='directional'>
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -1</direction>
    </light>
    
    <!-- Model placement -->
    <model name='my_humanoid_robot'>
      <include>
        <uri>model://humanoid_description</uri>
      </include>
      <pose>0 0 1.0 0 0 0</pose>
    </model>
    
    <!-- Static objects -->
    <model name='table'>
      <static>true</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

#### Model Creation

Humanoid robot models in Gazebo require detailed URDF/SDF descriptions with:

- Accurate kinematic chains
- Proper joint limits and dynamics
- Collision and visual geometries
- Inertial properties for each link

### Sensor Simulation in Gazebo

Gazebo provides realistic sensor simulation crucial for humanoid robot development:

#### LiDAR Simulation

```xml
<sensor name='lidar_front' type='ray'>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <pose>0.2 0 0.1 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name='lidar_controller' filename='libgazebo_ros_ray_sensor.so'>
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>~/out:=front_scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

#### Depth Camera Simulation

```xml
<sensor name='depth_camera' type='depth'>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name='depth_cam'>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <plugin name='camera_controller' filename='libgazebo_ros_openni_kinect.so'>
    <ros>
      <namespace>/humanoid/camera</namespace>
      <remapping>~/rgb/image_raw:=image_color</remapping>
      <remapping>~/depth/image_raw:=image_depth</remapping>
      <remapping>~/points:=points</remapping>
    </ros>
    <baseline>0.2</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
  </plugin>
</sensor>
```

#### IMU Simulation

```xml
<sensor name='imu_sensor' type='imu'>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <pose>0 0 0 0 0 0</pose>
  <imu>
    <angular_velocity>
      <x>
        <noise type='gaussian'>
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </x>
      <y>
        <noise type='gaussian'>
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </y>
      <z>
        <noise type='gaussian'>
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type='gaussian'>
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </x>
      <y>
        <noise type='gaussian'>
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type='gaussian'>
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name='imu_plugin' filename='libgazebo_ros_imu.so'>
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>~/out:=imu/data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
  </plugin>
</sensor>
```

### ROS 2 Integration with Gazebo

Gazebo integrates seamlessly with ROS 2 through plugins:

#### Spawning Models

```bash
# Spawn robot model from URDF
ros2 run gazebo_ros spawn_entity.py -entity my_humanoid -file /path/to/robot.urdf -x 0 -y 0 -z 1.0

# Spawn with namespace
ros2 run gazebo_ros spawn_entity.py -entity my_humanoid -topic robot_description -robot_namespace /humanoid
```

#### Control Interface

Gazebo supports ROS 2 control through the gazebo_ros_control plugin:

```xml
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <parameters>$(find my_humanoid_description)/config/controllers.yaml</parameters>
  </plugin>
</gazebo>
```

Controller configuration:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100
    use_sim_time: true

joint_state_broadcaster:
  type: joint_state_broadcaster/JointStateBroadcaster

position_controllers:
  type: position_controllers/JointGroupPositionController
  joints:
    - hip_joint
    - knee_joint
    - ankle_joint
```

## Unity Simulation Environment

Unity provides high-fidelity rendering and advanced physics simulation, making it ideal for applications requiring photorealistic visuals and complex environment modeling.

### High-Fidelity Rendering

Unity's rendering pipeline offers several advantages for humanoid robot simulation:

- **Physically-Based Rendering (PBR)**: Accurate material representation
- **Global Illumination**: Realistic lighting simulation
- **Advanced Shaders**: Custom visual effects for sensors
- **Post-Processing Effects**: Depth of field, motion blur, color grading

### Unity Robotics Simulation Framework

Unity's robotics simulation framework includes:

#### URDF Importer

The Unity URDF Importer converts ROS URDF files into Unity GameObjects:

```csharp
using Unity.Robotics.URDFImport;
using UnityEngine;

public class HumanoidSetup : MonoBehaviour
{
    [SerializeField] private string urdfPath = "Assets/Robots/humanoid.urdf";
    
    void Start()
    {
        // Load URDF model
        var robot = URDFRobotExtensions.LoadFromURDF(urdfPath);
        robot.SetMotionType(MotionType.Dynamic);
        
        // Configure joints
        ConfigureJoints(robot);
    }
    
    void ConfigureJoints(URDFRobot robot)
    {
        foreach (var joint in robot.GetComponentsInChildren<ArticulationBody>())
        {
            // Set joint limits and drive parameters
            var drive = joint.linearXDrive;
            drive.forceLimit = 100f;
            joint.linearXDrive = drive;
        }
    }
}
```

#### Sensor Simulation

Unity provides realistic sensor simulation through its physics and rendering systems:

##### Depth Camera Implementation

```csharp
using UnityEngine;
using Unity.Robotics.Sensors;

public class DepthCamera : MonoBehaviour
{
    [Header("Depth Camera Settings")]
    [Range(0.1f, 100f)] public float minDistance = 0.1f;
    [Range(0.1f, 100f)] public float maxDistance = 10.0f;
    
    private Camera cam;
    private RenderTexture depthTexture;
    
    void Start()
    {
        cam = GetComponent<Camera>();
        SetupDepthRendering();
    }
    
    void SetupDepthRendering()
    {
        depthTexture = new RenderTexture(640, 480, 24);
        cam.targetTexture = depthTexture;
        cam.depthTextureMode = DepthTextureMode.Depth;
    }
    
    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Apply depth shader
        Graphics.Blit(source, destination, GetDepthMaterial());
    }
    
    Material GetDepthMaterial()
    {
        // Return custom depth visualization material
        return Resources.Load<Material>("DepthVisualization");
    }
}
```

##### LiDAR Simulation

```csharp
using System.Collections.Generic;
using UnityEngine;

public class LidarSimulator : MonoBehaviour
{
    [Header("LiDAR Settings")]
    public int rayCount = 360;
    [Range(0.1f, 30f)] public float maxRange = 10f;
    public LayerMask detectionLayers = -1;
    
    private List<float> distances = new List<float>();
    
    void Update()
    {
        ScanEnvironment();
        PublishData();
    }
    
    void ScanEnvironment()
    {
        distances.Clear();
        float angleStep = 360f / rayCount;
        
        for (int i = 0; i < rayCount; i++)
        {
            float angle = Mathf.Deg2Rad * (transform.eulerAngles.y + i * angleStep);
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange, detectionLayers))
            {
                distances.Add(hit.distance);
            }
            else
            {
                distances.Add(maxRange);
            }
        }
    }
    
    void PublishData()
    {
        // Convert to ROS message format and publish
        // Implementation depends on ROS# or similar bridge
    }
}
```

### ROS 2 Integration with Unity

Unity integrates with ROS 2 through several methods:

#### ROS# Communication Library

```csharp
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor;

public class UnityRosBridge : MonoBehaviour
{
    private RosSocket rosSocket;
    
    void Start()
    {
        // Connect to ROS bridge
        WebSocketProtocols protocol = new WebSocketProtocols();
        protocol.Protocol = WebSocketProtocol.VanillaWebSocket;
        
        rosSocket = new RosSocket(protocol, "ws://localhost:9090");
        
        // Subscribe to joint commands
        rosSocket.Subscribe<JointState>("/humanoid/joint_commands", JointCommandCallback);
        
        // Publish sensor data
        InvokeRepeating(nameof(PublishSensorData), 0.0f, 0.1f);
    }
    
    void JointCommandCallback(JointState jointState)
    {
        // Update Unity robot model based on ROS commands
        UpdateRobotJoints(jointState.position);
    }
    
    void PublishSensorData()
    {
        // Publish IMU data
        Imu imuMsg = new Imu();
        // Populate with Unity sensor data
        rosSocket.Publish("/humanoid/imu/data", imuMsg);
    }
}
```

## Best Practices for Digital Twin Development

### Simulation Accuracy

- **Parameter Tuning**: Match simulated parameters to real robot characteristics
- **Validation Experiments**: Compare simulation and real robot behavior
- **Domain Randomization**: Introduce variations to improve real-world transfer

### Performance Optimization

- **Level of Detail (LOD)**: Adjust detail based on distance from sensors
- **Physics Simplification**: Use simplified collision meshes for performance
- **Sensor Resolution**: Balance accuracy with computational requirements

### Cross-Platform Consistency

- **Shared Assets**: Maintain consistent models between Gazebo and Unity
- **Standardized Interfaces**: Use common ROS message types
- **Validation Protocols**: Establish procedures to verify simulation accuracy

## Conclusion

Digital twin technology through Gazebo and Unity provides essential capabilities for humanoid robot development. Gazebo excels in physics accuracy and ROS integration, making it ideal for control algorithm development and validation. Unity offers superior rendering quality and flexibility for high-fidelity simulation and visualization.

The combination of these simulation environments allows developers to validate algorithms across different fidelity levels, from physics-focused Gazebo simulations to photorealistic Unity environments. Proper integration with ROS 2 ensures seamless transfer of code between simulation and real hardware, accelerating the development cycle for Physical AI systems.

As we advance to subsequent chapters, we'll explore how these simulation environments integrate with perception systems, control algorithms, and learning frameworks to create truly intelligent humanoid robots.