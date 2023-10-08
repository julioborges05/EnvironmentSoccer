from math import atan2, fabs, pi, fmod


def smallestAngleDiff(target, source):
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


def controller(field, objectives):
    """
        Basic PID controller that sets the speed of each motor
        sends robot to objective coordinate
        Courtesy of RoboCin
    """

    speeds = [{"index": i} for i in range(3)]
    our_bots = field["our_bots"]

    # for each bot
    for i, s in enumerate(speeds):
        Kp = 20
        Kd = 2.5

        try:
            controller.lastError
        except AttributeError:
            controller.lastError = 0

        right_motor_speed = 0
        left_motor_speed = 0

        objective = objectives[i]
        our_bot = our_bots[i]

        angle_rob = our_bot.a

        angle_obj = atan2(objective.y - our_bot.y,
                          objective.x - our_bot.x)

        error = smallestAngleDiff(angle_rob, angle_obj)

        reversed = False
        if fabs(error) > pi / 2.0 + pi / 20.0:
            reversed = True
            angle_rob = convert_angle(angle_rob + pi)
            error = smallestAngleDiff(angle_rob, angle_obj)

        # set motor speed based on error and K constants
        error_speed = (Kp * error) + (Kd * (error - controller.lastError))

        controller.lastError = error

        baseSpeed = 30

        # normalize
        error_speed = error_speed if error_speed < baseSpeed else baseSpeed
        error_speed = error_speed if error_speed > -baseSpeed else -baseSpeed

        if error_speed > 0:
            left_motor_speed = baseSpeed
            right_motor_speed = baseSpeed - error_speed
        else:
            left_motor_speed = baseSpeed + error_speed
            right_motor_speed = baseSpeed

        if reversed:
            if error_speed > 0:
                left_motor_speed = -baseSpeed + error_speed
                right_motor_speed = -baseSpeed
            else:
                left_motor_speed = -baseSpeed
                right_motor_speed = -baseSpeed - error_speed

        s["left"] = left_motor_speed
        s["right"] = right_motor_speed
    return speeds

def goal_keeper_controller(objective_x, objective_y, robot_angle, robot_x, robot_y):
    """
        Basic PID controller that sets the speed of each motor
        sends robot to objective coordinate
        Courtesy of RoboCin
    """

    # for each bot

    Kp = 20
    Kd = 2.5

    try:
        controller.lastError
    except AttributeError:
        controller.lastError = 0

    right_motor_speed = 0
    left_motor_speed = 0

    angle_rob = robot_angle

    angle_obj = atan2(objective_y - robot_y,
                      objective_x - robot_x)

    error = smallestAngleDiff(angle_rob, angle_obj)

    reversed = False
    if fabs(error) > pi / 2.0 + pi / 20.0:
        reversed = True
        angle_rob = convert_angle(angle_rob + pi)
        error = smallestAngleDiff(angle_rob, angle_obj)

    # set motor speed based on error and K constants
    error_speed = (Kp * error) + (Kd * (error - controller.lastError))

    controller.lastError = error

    baseSpeed = 30

    # normalize
    error_speed = error_speed if error_speed < baseSpeed else baseSpeed
    error_speed = error_speed if error_speed > -baseSpeed else -baseSpeed

    if error_speed > 0:
        left_motor_speed = baseSpeed
        right_motor_speed = baseSpeed - error_speed
    else:
        left_motor_speed = baseSpeed + error_speed
        right_motor_speed = baseSpeed

    if reversed:
        if error_speed > 0:
            left_motor_speed = -baseSpeed + error_speed
            right_motor_speed = -baseSpeed
        else:
            left_motor_speed = -baseSpeed
            right_motor_speed = -baseSpeed - error_speed

    speed_v1 = left_motor_speed
    speed_v2 = right_motor_speed
    return speed_v1, speed_v2