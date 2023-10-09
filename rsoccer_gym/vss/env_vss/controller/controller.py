from math import atan2, fabs, pi, fmod, atan, tan
from rsoccer_gym.Entities.Ball import Ball
from rsoccer_gym.vss.env_vss.controller.entity import Entity
from rsoccer_gym.vss.env_vss.controller.triangle import Triangle


def smallest_angle_diff(target, source):
    """Gets the smallest angle between two points in a arch"""
    a = fmod(target + 2 * pi, 2 * pi) - fmod(source + 2 * pi, 2 * pi)

    if a > pi:
        a -= 2 * pi
    else:
        if a < -pi:
            a += 2 * pi

    return a


def convert_angle(a) -> float:
    """
    Converts the angle from full radians to
    -Pi to Pi radians range
    """
    try:
        if a > 0:
            if a > pi:
                return a - 2 * pi
        else:
            if a < -pi:
                return a + 2 * pi
        return a

    except TypeError:
        return 0


def set_goal_keeper_coordinates(ball: Ball):
    goal_keeper_robot = Entity()
    goal_keeper_robot.x = -0.75

    try:
        is_ball_going_down = ball.v_y < 0
        big_triangle = get_big_triangle_values(is_ball_going_down, ball)
        small_triangle = get_small_triangle_values(big_triangle, ball)

        goal_keeper_robot.y = get_goal_keeper_vertical_value(is_ball_going_down, small_triangle.vertical_value, ball)
        return goal_keeper_robot.x, goal_keeper_robot.y
    except ZeroDivisionError:
        return -0.75, 0


def get_big_triangle_values(is_ball_going_down, ball: Ball):

    ball_velocity_angle = atan(ball.v_x / ball.v_y)

    big_triangle_vertical_value = ball.y + (0.20 if is_ball_going_down else -0.20)
    big_triangle_horizontal_value = tan(ball_velocity_angle) * big_triangle_vertical_value
    return Triangle(big_triangle_horizontal_value, big_triangle_vertical_value)


def get_small_triangle_values(big_triangle: Triangle, ball: Ball):
    goal_horizontal_position = -0.75
    small_triangle_horizontal_value = goal_horizontal_position - (ball.x - big_triangle.horizontal_value)
    small_triangle_vertical_value = (small_triangle_horizontal_value * big_triangle.vertical_value) / big_triangle.horizontal_value

    return Triangle(small_triangle_horizontal_value, small_triangle_vertical_value)


def get_goal_keeper_vertical_value(is_ball_going_down, small_triangle_vertical_value, ball: Ball):
    is_ball_going_to_other_goal_in_horizontal_direction = ball.v_x > 0
    robot_vertical_position = ball.y if is_ball_going_to_other_goal_in_horizontal_direction \
        else (small_triangle_vertical_value - 0.20) if is_ball_going_down else (small_triangle_vertical_value + 0.20)

    if robot_vertical_position > 0.20:
        return 0.165
    if robot_vertical_position < -0.20:
        return -0.165
    return robot_vertical_position


def goal_keeper_controller(objective_x, objective_y, robot_angle, robot_x, robot_y):
    """
        Basic PID controller that sets the speed of each motor
        sends robot to objective coordinate
        Courtesy of RoboCin
    """

    Kp = 20
    Kd = 2.5

    try:
        goal_keeper_controller.lastError
    except AttributeError:
        goal_keeper_controller.lastError = 0

    right_motor_speed = 0
    left_motor_speed = 0

    angle_rob = robot_angle

    angle_obj = atan2(objective_y - robot_y,
                      objective_x - robot_x)

    error = smallest_angle_diff(angle_rob, angle_obj)

    is_reversed = False

    if fabs(error) > pi / 2.0 + pi / 20.0:
        is_reversed = True
        angle_rob = convert_angle(angle_rob + pi)
        error = smallest_angle_diff(angle_rob, angle_obj)

    # set motor speed based on error and K constants
    error_speed = (Kp * error) + (Kd * (error - goal_keeper_controller.lastError))
    goal_keeper_controller.lastError = error
    base_speed = 30

    # normalize
    error_speed = error_speed if error_speed < base_speed else base_speed
    error_speed = error_speed if error_speed > -base_speed else -base_speed

    if error_speed > 0:
        left_motor_speed = base_speed
        right_motor_speed = base_speed - error_speed
    else:
        left_motor_speed = base_speed + error_speed
        right_motor_speed = base_speed

    if is_reversed:
        if error_speed > 0:
            left_motor_speed = -base_speed + error_speed
            right_motor_speed = -base_speed
        else:
            left_motor_speed = -base_speed
            right_motor_speed = -base_speed - error_speed

    return left_motor_speed, right_motor_speed
