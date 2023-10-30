from math import atan


class Triangle:
    def __init__(self, horizontal_value, vertical_value):
        self.horizontal_value = horizontal_value
        self.vertical_value = vertical_value
        self.vertical_hypotenuse_angle = self.get_vertical_hypotenuse_angle()

    def thales_theorem_vertical_value(self, horizontal_value):
        try:
            return (horizontal_value * self.vertical_value) / self.horizontal_value
        except ZeroDivisionError:
            return 0

    def thales_theorem_horizontal_value(self, vertical_value):
        try:
            return (self.horizontal_value * vertical_value) / self.vertical_value
        except ZeroDivisionError:
            return 0

    def get_vertical_hypotenuse_angle(self):
        try:
            return atan(self.horizontal_value / self.vertical_value)
        except ZeroDivisionError:
            return 0
