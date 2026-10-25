"""Small ROS client that can run inside Perception without copying Python tools."""
import json


def reload_model():
    import rclpy
    from std_srvs.srv import Trigger
    import yaml
    rclpy.init()
    node = rclpy.create_node('bin_model_deployment_cli')
    try:
        client = node.create_client(Trigger, 'bin_model/reload')
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError('Perception bin_model/reload service is unavailable')
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=60)
        if not future.done() or future.result() is None:
            raise RuntimeError('Perception model reload timed out')
        reply = future.result()
        if not reply.success:
            raise RuntimeError(reply.message)
        return yaml.safe_load(reply.message)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    print(json.dumps(reload_model()))
