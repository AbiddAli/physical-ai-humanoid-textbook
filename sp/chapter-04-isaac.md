# Chapter 4: NVIDIA Isaac Sim & Isaac ROS for Humanoid Robots

## Introduction to NVIDIA Isaac Platform

The NVIDIA Isaac platform represents a comprehensive solution for robotics development, combining Isaac Sim for high-fidelity simulation with Isaac ROS for accelerated perception and control. For humanoid robots, Isaac provides the computational power and realism needed to develop complex behaviors in virtual environments before deployment to physical hardware.

Isaac Sim leverages NVIDIA's expertise in graphics and AI to deliver:
- Physically accurate simulation with NVIDIA PhysX
- Photorealistic rendering using RTX technology
- Synthetic data generation for training perception systems
- Domain randomization for robust algorithm development

## Isaac Sim Setup and Physics Simulation

### Installation and Prerequisites

Isaac Sim requires specific hardware and software prerequisites:

- **GPU**: NVIDIA RTX series GPU with CUDA support (RTX 3080 or higher recommended)
- **OS**: Ubuntu 20.04 or 22.04 LTS
- **RAM**: 32GB minimum, 64GB recommended
- **CUDA**: 11.8 or later
- **Container Runtime**: Docker with nvidia-docker2

Installation typically involves downloading the Omniverse Isaac Sim application from NVIDIA Developer Zone and setting up the container runtime environment.

### Physics Simulation with PhysX

Isaac Sim uses NVIDIA PhysX as its physics engine, providing advanced capabilities for humanoid robot simulation:

- **Multi-body Dynamics**: Accurate simulation of complex articulated systems
- **Contact Response**: Realistic friction and collision handling
- **Soft Body Simulation**: Deformable object interactions
- **Fluid Simulation**: Environmental interactions with liquids and granular materials

Configuring physics parameters for humanoid robots:

```python
import omni
from pxr import UsdPhysics, Sdf, Gf

def configure_physics_properties(stage, robot_prim_path):
    """Configure physics properties for humanoid robot"""
    
    # Set up rigid body properties
    UsdPhysics.RigidBodyAPI.Apply(robot_prim_path)
    
    # Configure collision approximation
    collision_api = UsdPhysics.CollisionAPI.Apply(robot_prim_path)
    collision_api.CreateApproximationAttr().Set("convexDecomposition")
    
    # Set up articulation
    articulation_root_api = UsdPhysics.ArticulationRootAPI.Apply(robot_prim_path)
    articulation_root_api.GetEnabledSelfCollisionsAttr().Set(False)
    
    # Configure solver properties
    phys_scene_path = Sdf.Path("/physicsScene")
    phys_scene = UsdPhysics.Scene.Define(stage, phys_scene_path)
    phys_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    phys_scene.CreateGravityMagnitudeAttr(9.81)
    
    # Set solver iterations for stability
    phys_scene.CreatePositionIterationCountAttr(8)
    phys_scene.CreateVelocityIterationCountAttr(4)

def configure_joint_properties(joint_prim, stiffness=1e5, damping=1e3, limit=0.5):
    """Configure joint properties for humanoid robot"""
    
    # Apply joint API
    joint_api = UsdPhysics.JointAPI.Apply(joint_prim)
    
    # Set drive properties for position control
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive_api.CreateStiffnessAttr(stiffness)
    drive_api.CreateDampingAttr(damping)
    
    # Set joint limits
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, "angular")
    limit_api.CreateLowerAttr(-limit)
    limit_api.CreateUpperAttr(limit)
```

### Scene Building and Environment Creation

Isaac Sim uses USD (Universal Scene Description) as its scene format, allowing for complex environment creation:

```python
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.semantics import add_semantic_label

def create_humanoid_environment():
    """Create a humanoid-friendly environment in Isaac Sim"""
    
    # Add ground plane
    add_reference_to_stage(
        usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Grid/default_environment.usd",
        prim_path="/World/ground_plane"
    )
    
    # Create furniture and obstacles
    create_prim(
        prim_path="/World/table",
        prim_type="Cylinder",
        position=[2.0, 0.0, 0.4],
        attributes={"radius": 0.5, "height": 0.8}
    )
    
    # Add semantic labels for perception training
    add_semantic_label(prim_path="/World/table", semantic_label="furniture")
    
    # Create humanoid robot
    add_reference_to_stage(
        usd_path="path/to/humanoid_robot.usd",
        prim_path="/World/HumanoidRobot"
    )

def setup_lighting_and_cameras():
    """Configure realistic lighting and sensors"""
    
    # Add dome light for environment lighting
    create_prim(
        prim_path="/World/DomeLight",
        prim_type="DomeLight",
        attributes={"color": (0.2, 0.2, 0.2)}
    )
    
    # Add directional light for shadows
    create_prim(
        prim_path="/World/DirectionalLight",
        prim_type="DistantLight",
        position=[0, 0, 10],
        attributes={"color": (0.8, 0.8, 0.8), "intensity": 3000}
    )
```

## Photorealistic Rendering and Synthetic Data Generation

### RTX Rendering Pipeline

Isaac Sim leverages NVIDIA RTX technology for photorealistic rendering:

- **Ray Tracing**: Global illumination and accurate reflections
- **Deep Learning Denoising**: Real-time path tracing with reduced noise
- **Material System**: Physically-based materials with texture streaming
- **Volume Rendering**: Atmospheric effects and volumetric lighting

### Synthetic Data Generation

Synthetic data generation is crucial for training perception systems without real-world data collection:

```python
import omni
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.synthetic_utils.sensors import Camera
import numpy as np

def setup_synthetic_data_generation():
    """Configure synthetic data generation pipeline"""
    
    # Create camera for data capture
    camera = Camera(
        prim_path="/World/HumanoidRobot/head_camera",
        frequency=30,
        resolution=(640, 480)
    )
    
    # Enable various data streams
    camera.add_data_output("rgb", "rgba")  # Color images
    camera.add_data_output("depth", "distance_to_image_plane")  # Depth maps
    camera.add_data_output("semantic_segmentation", "semantic_segmentation_id")  # Semantic segmentation
    camera.add_data_output("instance_segmentation", "instance_segmentation_id")  # Instance segmentation
    camera.add_data_output("bounding_box_2d_tight", "bbox_2d_tight")  # Bounding boxes
    
    return camera

def domain_randomization_pipeline():
    """Apply domain randomization for robust perception"""
    
    # Randomize lighting conditions
    def randomize_lighting():
        lights = ["/World/DomeLight", "/World/DirectionalLight"]
        for light_path in lights:
            # Randomize color temperature
            color_temp = np.random.uniform(5000, 8000)
            # Convert to RGB approximation
            rgb_color = color_temperature_to_rgb(color_temp)
            omni.kit.commands.execute(
                "ChangePropertyCommand",
                prop_path=Sdf.Path(f"{light_path}.inputs:color"),
                value=rgb_color
            )
    
    # Randomize textures and materials
    def randomize_materials():
        # Cycle through material variations
        material_paths = get_material_paths()
        for mat_path in material_paths:
            # Apply random texture variations
            texture_variation = np.random.choice(["wood", "metal", "plastic"])
            apply_texture_variation(mat_path, texture_variation)
    
    # Randomize object positions
    def randomize_object_positions():
        # Move objects within defined bounds
        objects = get_movable_objects()
        for obj_path in objects:
            new_pos = np.random.uniform([-5, -5, 0], [5, 5, 2])
            set_object_position(obj_path, new_pos)
    
    return {
        "lighting": randomize_lighting,
        "materials": randomize_materials,
        "positions": randomize_object_positions
    }

def generate_training_dataset(num_samples=10000):
    """Generate synthetic dataset for perception training"""
    
    camera = setup_synthetic_data_generation()
    randomizers = domain_randomization_pipeline()
    
    for i in range(num_samples):
        # Apply randomizations
        for randomizer_func in randomizers.values():
            randomizer_func()
        
        # Capture frame
        frame_data = camera.get_frame()
        
        # Save data with annotations
        save_training_sample(frame_data, sample_id=i)
        
        # Step simulation
        omni.timeline.get_timeline_interface().step(1)
```

## Visual SLAM for Perception and Navigation

### Isaac ROS VSLAM Components

Isaac ROS provides optimized implementations of visual SLAM algorithms:

- **Stereo Visual Odometry**: Real-time pose estimation from stereo cameras
- **Feature Tracking**: GPU-accelerated feature detection and matching
- **Map Building**: Incremental construction of 3D maps
- **Loop Closure**: Recognition of previously visited locations

### Setting Up VSLAM in Isaac Sim

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class HumanoidVslamNode(Node):
    def __init__(self):
        super().__init__('humanoid_vslam')
        
        # Subscribe to camera feeds
        self.left_cam_sub = self.create_subscription(
            Image,
            '/humanoid/stereo/left/image_rect_color',
            self.left_image_callback,
            10
        )
        
        self.right_cam_sub = self.create_subscription(
            Image,
            '/humanoid/stereo/right/image_rect_color',
            self.right_image_callback,
            10
        )
        
        # Subscribe to camera info
        self.left_info_sub = self.create_subscription(
            CameraInfo,
            '/humanoid/stereo/left/camera_info',
            self.left_info_callback,
            10
        )
        
        self.right_info_sub = self.create_subscription(
            CameraInfo,
            '/humanoid/stereo/right/camera_info',
            self.right_info_callback,
            10
        )
        
        # Publish odometry
        self.odom_pub = self.create_publisher(Odometry, '/humanoid/odom', 10)
        
        # Initialize VSLAM pipeline
        self.initialize_vslam_pipeline()
    
    def initialize_vslam_pipeline(self):
        """Initialize Isaac ROS VSLAM pipeline"""
        # Configure stereo rectification
        self.stereo_rectifier = StereoRectifier()
        
        # Initialize feature detector
        self.feature_detector = FeatureDetector(
            max_features=2000,
            pyramid_levels=4
        )
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            baseline=0.2,  # Baseline in meters
            focal_length=525.0  # Focal length in pixels
        )
    
    def left_image_callback(self, msg):
        """Process left camera image"""
        # Convert ROS image to OpenCV
        cv_image = self.ros_to_cv2(msg)
        
        # Extract features
        features = self.feature_detector.detect(cv_image)
        
        # Store for stereo matching
        self.left_features = features
    
    def right_image_callback(self, msg):
        """Process right camera image and compute pose"""
        if hasattr(self, 'left_features'):
            cv_image = self.ros_to_cv2(msg)
            
            # Match features between left and right
            matches = self.match_features(self.left_features, cv_image)
            
            # Compute disparity and 3D points
            disparities = self.compute_disparities(matches)
            points_3d = self.triangulate_points(disparities)
            
            # Estimate pose
            pose = self.pose_estimator.estimate_pose(points_3d)
            
            # Publish odometry
            self.publish_odometry(pose)
    
    def publish_odometry(self, pose):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'humanoid/base_link'
        
        # Set position and orientation
        odom_msg.pose.pose.position.x = pose[0]
        odom_msg.pose.pose.position.y = pose[1]
        odom_msg.pose.pose.position.z = pose[2]
        
        # Convert rotation matrix to quaternion
        quat = self.rotation_matrix_to_quaternion(pose[3:])
        odom_msg.pose.pose.orientation = quat
        
        self.odom_pub.publish(odom_msg)
```

## Nav2 Path Planning for Bipedal Humanoid Movement

### Humanoid-Specific Navigation Considerations

Humanoid robots present unique challenges for navigation:

- **Footstep Planning**: Need to plan discrete foot placements
- **Balance Constraints**: Maintain center of mass within support polygon
- **Dynamic Obstacles**: Account for moving humans in shared spaces
- **Terrain Adaptation**: Handle uneven surfaces and obstacles

### Configuring Nav2 for Humanoid Robots

```yaml
# humanoid_nav2_config.yaml
bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: map
    robot_base_frame: base_link
    bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.2
      yaw_goal_tolerance: 0.1
      stateful: True

controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 32
      model_dt: 0.1
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.3
      vx_max: 0.5
      vx_min: -0.2
      vy_max: 0.2
      vy_min: -0.2
      wz_max: 0.5
      wz_min: -0.5
      goal_dist_tol: 0.2
      xy_goal_tol: 0.2
      trans_stopped_vel: 0.25
      short_circuit_iterations: 5
      obstacle_cost_weight: 1.0
      goal_cost_weight: 1.0
      reference_cost_weight: 1.0
      curvature_cost_weight: 0.0
      forward_cost_weight: -0.1
      collision_cost_weight: 100.0
      cost_per_turning_degree: 0.0
      cost_per_steering_change: 0.0
      motion_model: "DiffDrive"
      visualize: false

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Humanoid-specific radius
      plugins: ["obstacle_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /humanoid/laser_scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: true
      robot_radius: 0.3
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
```

### Footstep Planner Integration

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from builtin_interfaces.msg import Duration
from tf2_ros import TransformListener, Buffer

class HumanoidFootstepPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_footstep_planner')
        
        # Navigation action client
        self.nav_action_client = ActionClient(
            self, 
            NavigateToPose, 
            'navigate_to_pose'
        )
        
        # TF buffer for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Footstep publisher
        self.footstep_pub = self.create_publisher(
            HumanoidFootsteps, 
            '/humanoid/footsteps', 
            10
        )
        
        # Initialize footstep planner
        self.footstep_generator = FootstepGenerator(
            step_length=0.3,
            step_width=0.2,
            max_yaw_change=0.3
        )
    
    def plan_navigation_with_footsteps(self, goal_pose):
        """Plan navigation considering footstep constraints"""
        
        # Get current robot pose
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().error(f'Transform lookup failed: {e}')
            return False
        
        current_pose = PoseStamped()
        current_pose.header.frame_id = 'map'
        current_pose.pose.position.x = transform.transform.translation.x
        current_pose.pose.position.y = transform.transform.translation.y
        current_pose.pose.orientation = transform.transform.rotation
        
        # Generate footstep plan
        footsteps = self.footstep_generator.generate_path(
            start_pose=current_pose.pose,
            goal_pose=goal_pose.pose
        )
        
        # Check feasibility of footstep plan
        if not self.validate_footstep_plan(footsteps):
            self.get_logger().warn('Footstep plan not feasible')
            return False
        
        # Execute navigation
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        goal_msg.behavior_tree_id = ""
        
        self.nav_action_client.wait_for_server()
        future = self.nav_action_client.send_goal_async(goal_msg)
        
        # Publish footsteps for balance controller
        self.publish_footsteps(footsteps)
        
        return future
    
    def validate_footstep_plan(self, footsteps):
        """Validate footstep plan for balance and collision"""
        
        for i, step in enumerate(footsteps):
            # Check for collisions
            if self.check_collision(step):
                return False
            
            # Check balance constraints
            if i > 0:  # Need previous step for balance check
                if not self.check_balance_constraint(footsteps[i-1], step):
                    return False
        
        return True
    
    def publish_footsteps(self, footsteps):
        """Publish footsteps for balance controller"""
        footstep_msg = HumanoidFootsteps()
        footstep_msg.header.stamp = self.get_clock().now().to_msg()
        footstep_msg.header.frame_id = 'map'
        
        for step in footsteps:
            footstep_pose = PoseStamped()
            footstep_pose.header.frame_id = 'map'
            footstep_pose.pose = step
            footstep_msg.footsteps.append(footstep_pose)
        
        self.footstep_pub.publish(footstep_msg)
```

## Reinforcement Learning for Robot Control

### Isaac Gym Environments

Isaac Gym provides GPU-accelerated environments for reinforcement learning:

```python
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext

class HumanoidRLEnv:
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.sim_params = self.cfg["sim"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_actions = cfg["env"]["numActions"]
        self.num_envs = cfg["env"]["numEnvs"]
        
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim(sim_device, graphics_device_id, headless)
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
        
        # Load humanoid asset
        asset_root = cfg["env"]["asset"]["assetRoot"]
        asset_file = cfg["env"]["asset"]["assetFileName"]
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.set_ang_vel_scale(2.5)
        asset_options.density = 1000
        asset_options.max_angular_velocity = 50
        asset_options.fix_base_link = False
        
        self.humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        
        # Create environments
        self._create_envs()
        
        # Initialize tensors
        self._initialize_tensors()
    
    def _create_envs(self):
        """Create simulation environments"""
        spacing = 5.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.envs = []
        for i in range(self.num_envs):
            # Create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            
            # Add humanoid to environment
            humanoid_pose = gymapi.Transform()
            humanoid_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            humanoid_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
            
            humanoid_handle = self.gym.create_actor(
                env, self.humanoid_asset, humanoid_pose, "humanoid", i, 1
            )
            
            # Configure DOF properties
            dof_props = self.gym.get_actor_dof_properties(env, humanoid_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:] = 0.0
            dof_props["damping"][:] = 0.1
            self.gym.set_actor_dof_properties(env, humanoid_handle, dof_props)
            
            self.envs.append(env)
    
    def _initialize_tensors(self):
        """Initialize tensor buffers"""
        # Actor root state tensor
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        
        # Dof state tensor
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Net contact force tensor
        net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_force_tensor)
        
        # Initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Action tensor
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
    
    def reset_idx(self, env_ids):
        """Reset environments"""
        positions = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        
        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = velocities
        
        # Reset root states
        root_pos = self.initial_root_states[env_ids].clone()
        root_pos[:, :3] += torch_rand_float(-0.5, 0.5, (len(env_ids), 3), device=self.device)
        root_pos[:, 7:9] = torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)
        
        self.root_states[env_ids] = root_pos
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(torch.arange(len(env_ids), device=self.device, dtype=torch.int32)), len(env_ids)
        )
        
        # Reset DOF states
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(torch.arange(len(env_ids)*2, device=self.device, dtype=torch.int32)), len(env_ids)*2
        )
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    def pre_physics_step(self, actions):
        """Apply actions before physics step"""
        self.actions = actions.clone()
        
        # Apply torque control
        torques = self.actions * 100.0  # Scale factor
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
    
    def post_physics_step(self):
        """Process simulation results after physics step"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        self.compute_observations()
        self.compute_rewards()
        
        # Reset environments that need resetting
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.progress_buf += 1
```

## Sim-to-Real Transfer Techniques

### Domain Randomization

Domain randomization is essential for transferring policies from simulation to reality:

```python
def apply_domain_randomization(env, epoch):
    """Apply domain randomization to improve sim-to-real transfer"""
    
    # Randomize physical parameters
    if epoch % 10 == 0:  # Apply every 10 epochs
        # Randomize mass properties
        mass_scale = np.random.uniform(0.8, 1.2)
        for i in range(env.num_envs):
            # Apply random mass scaling
            env.apply_mass_scaling(i, mass_scale)
        
        # Randomize friction coefficients
        friction_range = np.random.uniform(0.5, 1.5, size=env.num_envs)
        env.set_friction_coefficients(friction_range)
    
    # Randomize sensor noise
    sensor_noise = np.random.uniform(0.001, 0.01)
    env.set_sensor_noise_level(sensor_noise)
    
    # Randomize actuator delays
    actuator_delay = np.random.uniform(0.0, 0.02)  # 0-20ms delay
    env.set_actuator_delay(actuator_delay)

def system_identification_for_transfer():
    """Identify system parameters for better sim-to-real transfer"""
    
    # Collect real-world data
    real_data = collect_real_world_data()
    
    # Compare with simulation data
    sim_data = collect_simulation_data()
    
    # Identify parameter differences
    param_diffs = compare_system_responses(real_data, sim_data)
    
    # Update simulation parameters
    update_sim_params(param_diffs)
```

### Reality Gap Mitigation

Strategies to minimize the reality gap include:

- **System Identification**: Measuring real robot dynamics to tune simulation
- **Adaptive Control**: Adjusting control parameters based on real-time feedback
- **Online Fine-tuning**: Continuously updating policies during real-world deployment

## Best Practices and Considerations

### Performance Optimization

- **GPU Utilization**: Maximize GPU usage for physics and rendering
- **Batch Processing**: Process multiple environments in parallel
- **Efficient Rendering**: Use appropriate LOD and rendering settings

### Validation Strategies

- **Cross-validation**: Test policies across multiple simulation conditions
- **Ablation Studies**: Understand impact of different randomization elements
- **Real-world Testing**: Validate key behaviors on physical hardware

## Conclusion

NVIDIA Isaac Sim and Isaac ROS provide powerful tools for developing humanoid robots with advanced perception, navigation, and control capabilities. The combination of photorealistic rendering, physically accurate simulation, and GPU-accelerated processing enables the development of robust algorithms that can successfully transfer to real-world hardware.

The integration of VSLAM for perception, Nav2 for navigation adapted for humanoid constraints, and reinforcement learning for control creates a comprehensive development pipeline. As we move forward in this textbook, we'll explore how these simulation-based approaches integrate with real-world deployment and learning systems to create truly autonomous humanoid robots.