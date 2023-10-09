from math import atan


class Triangle:
    def __init__(self, horizontalValue, verticalValue):
        self.horizontalValue = horizontalValue
        self.verticalValue = verticalValue
        self.verticalHypotenuseAngle = self.__getVerticalHypotenuseAngle()

    def thalesTheoremVerticalValue(self, horizontalValue):
        try:
            return (horizontalValue * self.verticalValue) / self.horizontalValue
        except ZeroDivisionError:
            return 0

    def thalesTheoremHorizontalValue(self, verticalValue):
        try:
            return (self.horizontalValue * verticalValue) / self.verticalValue
        except ZeroDivisionError:
            return 0

    def __getVerticalHypotenuseAngle(self):
        try:
            return atan(self.horizontalValue / self.verticalValue)
        except ZeroDivisionError:
            return 0
