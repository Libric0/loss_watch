from typing import Callable
import math
from .color import Color
_palette = None


def set_palette(palette: Callable[[float], str] | list | str | None = None):
    global _palette

    if isinstance(palette, str):
        try:
            import seaborn as sns
        except ImportError:
            raise Exception(
                "Named palettes can only be used if seaborn is installed in your python environment. Run `pip install seaborn` to install it.")
        # Converting to SVG rgb format
        cmap = sns.color_palette(palette=palette, as_cmap=True)
        # Seaborn's color maps map from [0,1), 1 exclusive.
        # This means that values will wrap around completely if they get to 1.
        # Multiplying the position with 0.9999999 was the quickest fix.

        def palette_function(pos):
            pos = 1 - pos
            pos *= 0.99999999

            return f'#{str(Color(cmap(pos)[0], cmap(pos)[1], cmap(pos)[2]))}'
        _palette = palette_function
        return

    _palette = palette


def _standard_palette(position: float):
    return f'#{str(Color.from_hsl((1-position)*180, 1, .6))}'


def _list_color(palette: list[list[float]], position: float):
    if position == 0.0:
        return palette[0]
    if position == 1.0:
        return palette[-1]
    list_positon: float = position*len(palette) - 1
    # How much of the color left to the position is to be used
    percentage_left = list_positon - math.floor(list_positon)
    color = [c_l*percentage_left + c_r*(1-percentage_left) for c_l, c_r in zip(
        palette[math.floor(list_positon)], palette[math.floor(list_positon)+1])]
    return f'#{str(Color(color[0], color[1], color[2]))}'


def get_color_continuous(position: float) -> str:
    # Standard palette given by the library, 360 different hues but gets
    # crammed / hard to distinguish after 10
    if _palette is None:
        return _standard_palette(position)

    # Use user-defined palettes
    if isinstance(_palette, list):
        return _list_color(_palette, position)

    if isinstance(_palette, Callable):
        return _palette(position)
    import warnings
    warnings.warn(
        "The user-defined palette is not valid. Resorting to default.")
    return _standard_palette(position)


def get_contrasting_font_color(color_hex: str):
    color = Color.from_hex(color_hex)
    lightness = color.oklab["l"]
    if lightness < .7:
        return "ffffffff"
    else:
        return "000000ff"

def get_warning_font_color():
    return _standard_palette(1)