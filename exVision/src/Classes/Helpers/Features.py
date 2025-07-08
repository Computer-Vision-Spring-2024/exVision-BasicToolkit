import numpy as np


# base class to all features
class Feature:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.coords_x = None
        self.coords_y = None
        self.coeff = None

    def __call__(self, image_to_integrate):
        try:
            return np.sum(
                image_to_integrate[self.coords_y, self.coords_x] * self.coeffs
            )
        except IndexError as e:
            raise IndexError(str(e) + " in " + str(self))

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


# -----------------------------------------------------------------


class Feature2h(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        hw = width // 2
        self.coords_x = [x, x + hw, x, x + hw, x + hw, x + width, x + hw, x + width]
        self.coords_y = [y, y, y + height, y + height, y, y, y + height, y + height]
        self.coeffs = [1, -1, -1, 1, -1, 1, 1, -1]


# since we have two horizonal regions that get subtracted from each other


# -----------------------------------------------------------------------


class Feature2v(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        hh = height // 2  # two regions vertically
        self.coords_x = [x, x + width, x, x + width, x, x + width, x, x + width]
        self.coords_y = [y, y, y + hh, y + hh, y + hh, y + hh, y + height, y + height]
        self.coeffs = [-1, 1, 1, -1, 1, -1, -1, 1]


# -------------------------------------------------------------------------


class Feature3h(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        tw = width // 3  # third of width (we have 3 regions)
        self.coords_x = [
            x,
            x + tw,
            x,
            x + tw,
            x + tw,
            x + 2 * tw,
            x + tw,
            x + 2 * tw,
            x + 2 * tw,
            x + width,
            x + 2 * tw,
            x + width,
        ]
        self.coords_y = [
            y,
            y,
            y + height,
            y + height,
            y,
            y,
            y + height,
            y + height,
            y,
            y,
            y + height,
            y + height,
        ]
        self.coeffs = [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1]


# -------------------------------------------------------------------------


class Feature3v(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        th = height // 3  # 3 regions to be extracted
        self.coords_x = [
            x,
            x + width,
            x,
            x + width,
            x,
            x + width,
            x,
            x + width,
            x,
            x + width,
            x,
            x + width,
        ]
        self.coords_y = [
            y,
            y,
            y + th,
            y + th,
            y + th,
            y + th,
            y + 2 * th,
            y + 2 * th,
            y + 2 * th,
            y + 2 * th,
            y + height,
            y + height,
        ]
        self.coeffs = [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1]


# -----------------------------------------------------------------------


class Feature4(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        hw = width // 2
        hh = height // 2
        self.coords_x = [
            x,
            x + hw,
            x,
            x + hw,
            x + hw,
            x + width,
            x + hw,
            x + width,  # upper row
            x,
            x + hw,
            x,
            x + hw,
            x + hw,
            x + width,
            x + hw,
            x + width,
        ]
        self.coords_y = [
            y,
            y,
            y + hh,
            y + hh,
            y,
            y,
            y + hh,
            y + hh,
            y + hh,
            y + hh,
            y + height,
            y + height,
            y + hh,
            y + hh,
            y + height,
            y + height,
        ]
        self.coeffs = [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]
