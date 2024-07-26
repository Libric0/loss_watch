from __future__ import annotations
import math


def linear_srgb1_to_srgb1(channel: float) -> float:
    if channel <= 0.0031308:
        return channel*12.92
    return ((channel ** (1/2.4)) * 1.055) - 0.055


def srgb1_to_linear_srgb1(channel: float) -> float:
    if channel <= 0.0405:
        return channel/12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def clamp(channel: float) -> float:
    return max(0, min(1, channel))


class Color:
    def __init__(
            self,
            red: float,
            green: float,
            blue: float,
            alpha: float | None = 1
    ):
        self.red: float = clamp(red)
        self.green: float = clamp(green)
        self.blue: float = clamp(blue)
        self.alpha: float = clamp(alpha)

    @staticmethod
    def from_oklab(
            l: float,
            a: float,
            b: float,
            alpha: float = 1
    ) -> Color:
        l_ = l + 0.3963377774*a + 0.2158037573*b
        m_ = l - 0.1055613458*a - 0.0638541728*b
        s_ = l - 0.0894841775*a - 1.2914855480*b

        l_ **= 3
        m_ **= 3
        s_ **= 3

        red_linear = +4.0767416621*l_ - 3.3077115913*m_ + 0.2309699292*s_
        green_linear = -1.2684380046*l_ + 2.6097574011*m_ - 0.3413193965*s_
        blue_linear = -0.0041960863*l_ - 0.7034186147*m_ + 1.7076147010*s_

        red = linear_srgb1_to_srgb1(red_linear)
        green = linear_srgb1_to_srgb1(green_linear)
        blue = linear_srgb1_to_srgb1(blue_linear)

        return Color(red, green, blue, alpha)

    @property
    def oklab(self):
        red = srgb1_to_linear_srgb1(self.red)
        green = srgb1_to_linear_srgb1(self.green)
        blue = srgb1_to_linear_srgb1(self.blue)

        l = 0.4122214708*red + 0.5363325363*green + 0.0514459929*blue
        m = 0.2119034982*red + 0.6806995451*green + 0.1073969566*blue
        s = 0.0883024619*red + 0.2817188376*green + 0.6299787005*blue

        l_ = l**(1/3)
        m_ = m**(1/3)
        s_ = s**(1/3)

        return {
            "l": 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
            "a": 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
            "b": 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_,
            "alpha": self.alpha
        }

    @staticmethod
    def from_oklch(
        l,
        c,
        h,
        alpha: float = 1
    ) -> Color:
        h_rad = (2*math.pi*h)/360
        a = c*math.cos(h_rad)
        b = c*math.sin(h_rad)
        return Color.from_oklab(l, a, b, alpha)

    @property
    def oklch(self):
        oklab = self.oklab
        c = ((oklab["a"]**2) + (oklab["b"]**2)) ** (1/2)
        h_rad = math.atan2(oklab["b"], oklab["a"])
        h = 360*h_rad/(2*math.pi)
        return {
            "l": oklab["l"],
            "c": c,
            "h": h,
            "alpha": self.alpha
        }

    def from_hsl(h: float, s: float, l: float, alpha: float = 1) -> Color:
        h %= 360
        c = (1-abs((2*l) - 1))*s
        x = c*(1 - abs(((h/60) % 2) - 1))
        m = l - (c/2)
        if 0 <= h < 60:
            r_, g_, b_ = c, x, 0
        elif 60 <= h < 120:
            r_, g_, b_ = x, c, 0
        elif 120 <= h < 180:
            r_, g_, b_ = 0, c, x
        elif 180 <= h < 240:
            r_, g_, b_ = 0, x, c
        elif 240 <= h < 300:
            r_, g_, b_ = x, 0, c
        elif 300 <= h < 360:
            r_, g_, b_ = c, 0, x

        r = r_ + m
        g = g_ + m
        b = b_ + m

        return Color(r, g, b, alpha)

    @staticmethod
    def from_hex(hex: str) -> Color:
        # Removing any starting #
        if hex[0] == '#':
            hex_parsed = hex[1:]
        else:
            hex_parsed = hex
        if len(hex_parsed) != 6 and len(hex_parsed) != 8:
            raise ColorConvertException(f'A hex color has to have either 6 characters for RGB or 8 characters for RGBA, ignoring the #. The color {
                                        hex} has {len(hex_parsed)} characters')
        try:
            red = int(hex_parsed[0:2], 16)
            green = int(hex_parsed[2:4], 16)
            blue = int(hex_parsed[4:6], 16)
            alpha = None
            if len(hex_parsed) == 8:
                alpha = int(hex_parsed[6:8], 16)
        except ValueError as e:
            raise ColorConvertException(
                f'A hex color can only contain hexadecimal digits. The color {hex} does not.') from e
        return Color(red/255, green/255, blue/255, alpha/255)

    @property
    def hex(self) -> str:
        return f'{round(self.red*255):02X}{round(self.green*255):02X}{round(self.blue*255):02X}{round(self.alpha*255):02X}'

    def __str__(self):
        return f'{self.hex}'

    def __getitem__(self, item):
        if item not in [0, 1, 2, 3]:
            raise IndexError(
                f'Colors expect the index to be an integer between 0 and 3 inclusive, not {item}')
        match item:
            case 0:
                return self.red
            case 1:
                return self.blue
            case 2:
                return self.green
            case 3:
                return self.alpha

    def __dict__(self):
        return {
            "red": self.red,
            "green": self.green,
            "blue": self.blue,
            "alpha": self.alpha
        }


class ColorConvertException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
