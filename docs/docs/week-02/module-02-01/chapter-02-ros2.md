---
id: chapter-02-ros2
title: "Chapter 2: ROS 2 Fundamentals"
---

# Chapter 2: ROS 2 Fundamentals

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is the next-generation middleware framework designed for robotics applications. Unlike its predecessor, ROS 2 addresses critical issues such as real-time performance, security, and deployment in production environments. For Physical AI systems, particularly humanoid robots, ROS 2 provides the essential communication infrastructure that enables seamless interaction between perception, planning, and control modules.

ROS 2 implements a client library architecture that supports multiple programming languages while maintaining consistent messaging semantics. The framework utilizes Data Distribution Service (DDS) as its underlying communication layer, ensuring reliable message delivery even in distributed systems with multiple robots or computational units.

## ROS 2 Architecture

ROS 2's architecture is built around several core concepts:

- **Nodes**: Individual processes that perform computation
- **Topics**: Named buses for passing messages between nodes
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous goals with feedback and status information
- **Parameters**: Configuration values that can be changed at runtime

### Nodes and Communication

Nodes in ROS 2 are independent processes that communicate through a distributed communication mechanism. Each node can publish data to topics, subscribe to topics, provide services, or call services. The communication layer abstracts the underlying network topology, allowing nodes to communicate whether they're running on the same machine or distributed across a network.

```
Example node structure:
- Node: Perception Node
  - Publishes: /camera/image_raw
  - Subscribes: /robot/odometry
  - Provides: /object_detection
- Node: Controller Node
  - Subscribes: /sensor_data
  - Publishes: /motor_commands
```

### Topics and Message Passing

Topics in ROS 2 use a publish-subscribe pattern for asynchronous communication. Publishers send messages to topics without knowledge of subscribers, and subscribers receive messages from topics without knowledge of publishers. This loose coupling enables flexible system architectures where components can be added or removed without affecting others.

Key characteristics of topics:
- **Anonymous**: Publishers and subscribers are unaware of each other
- **Asynchronous**: Message delivery timing is not guaranteed
- **Typed**: Messages have defined schemas for type safety
- **Serialized**: Messages are converted to byte streams for transport

### Services and Actions

Services provide synchronous request-response communication, suitable for operations that return a result immediately. Actions extend this concept with support for long-running operations that provide feedback during execution.

Service characteristics:
- Request-response pattern
- Synchronous operation
- Single response per request
- Suitable for quick operations

Action characteristics:
- Long-running operations
- Feedback during execution
- Goal preemption capability
- Multiple status updates

## Quality of Service (QoS)

ROS 2 introduces Quality of Service profiles to handle different communication requirements. QoS settings allow nodes to specify reliability, durability, liveliness, and other communication characteristics.

### Reliability Policy
- **Reliable**: All messages are guaranteed to be delivered
- **Best Effort**: Messages may be lost but delivery is faster

### Durability Policy
- **Transient Local**: Publishers send old messages to new subscribers
- **Volatile**: New subscribers only receive future messages

### History Policy
- **Keep Last**: Store only the most recent messages
- **Keep All**: Store all messages (limited by resource constraints)

## Package Management

ROS 2 packages organize code, data, and configuration into reusable units. Each package contains a `package.xml` manifest and follows the File System Hierarchy Standard (FHS) for organization.

### Package Structure
```
my_robot_package/
├── CMakeLists.txt          # Build configuration
├── package.xml             # Package manifest
├── src/                    # Source code
├── include/                # Header files
├── launch/                 # Launch files
├── config/                 # Configuration files
├── scripts/                # Executable scripts
└── test/                   # Unit tests
```

### Build System

ROS 2 uses the `colcon` build system, which extends CMake for ROS packages. Colcon handles dependency resolution, parallel builds, and workspace management.

```
# Building a workspace
colcon build --packages-select my_robot_package
source install/setup.bash
```

## Client Libraries

ROS 2 supports multiple client libraries:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rclrs**: Rust client library
- **rclc**: C client library

Each client library provides consistent APIs across languages, enabling polyglot robotics systems.

## DDS Implementation

The Data Distribution Service (DDS) implementation provides the underlying communication infrastructure:

- **Fast DDS**: Default implementation in ROS 2 Humble
- **Cyclone DDS**: Alternative with lower resource usage
- **RTI Connext DDS**: Commercial implementation with additional features

## Time Management

ROS 2 provides sophisticated time management capabilities:

- **System Time**: Wall clock time from the operating system
- **Simulated Time**: Time from simulation environments
- **Steady Time**: Monotonically increasing time for intervals

```
# Using ROS time
rclcpp::Time start = this->get_clock()->now();
// Perform operation
rclcpp::Duration elapsed = this->get_clock()->now() - start;
```

## Security Features

ROS 2 includes security capabilities for production environments:

- **Authentication**: Verify identity of nodes and processes
- **Access Control**: Define permissions for topics and services
- **Encryption**: Protect message contents in transit
- **Audit Logging**: Track security-relevant events

## Testing and Debugging

ROS 2 provides comprehensive tools for testing and debugging:

- **rqt**: Graphical user interface for introspection
- **ros2 topic**: Command-line tools for topic inspection
- **ros2 service**: Service interaction tools
- **ros2 action**: Action interaction tools
- **rviz2**: 3D visualization environment

### Command Line Tools

```
# List available topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /sensor_data

# Call a service
ros2 service call /reset_robot std_srvs/Trigger

# List nodes
ros2 node list
```

## Integration with Physical AI Systems

For Physical AI applications, ROS 2 provides the communication backbone that connects:

- **Sensors**: Cameras, lidar, IMU, force/torque sensors
- **Controllers**: Joint controllers, trajectory planners
- **Perception**: Object detection, SLAM, scene understanding
- **Planning**: Path planning, manipulation planning
- **Actuation**: Motor commands, gripper control

This modular architecture enables rapid development and testing of Physical AI components while maintaining the flexibility to deploy on real hardware or simulation.

## Conclusion

ROS 2 represents a mature middleware solution for robotics applications, providing the foundation for complex Physical AI systems. Its emphasis on real-time performance, security, and production deployment makes it ideal for humanoid robotics where reliability and safety are paramount.

The architecture's flexibility allows for both centralized and distributed implementations, accommodating various system architectures from single-board computers to multi-computer clusters. As we progress through this textbook, we'll see how ROS 2's capabilities enable the sophisticated behaviors required for humanoid robots to interact safely and effectively with humans and their environments.