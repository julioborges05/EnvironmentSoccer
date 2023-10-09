from math import sqrt


class Entity:
    """
    Class used to determine the position, speed and direction
    of any entity on the field.
    """

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, a=None, va=0.0, index=None, comment=""):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.a = a
        self.va = va
        self.index = index
        self.comment = comment

    def distance_to(self, b):
        return sqrt((self.x - b.x) ** 2 + (self.y - b.y) ** 2)

    def __str__(self) -> str:
        return f"Entity: {self.x}, {self.y}, {self.a}"

    def __repr__(self) -> str:
        return self.__str__()