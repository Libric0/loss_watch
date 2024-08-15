from typing import Callable
import math
from .color import Color

# Color palettes
_color_palette: Callable[[float], str] | list | None = None


def set_color_palette(palette: Callable[[float], str] | list | str | None = None):
    global _color_palette

    if not isinstance(palette, (Callable, list, str, type(None))):
        raise TypeError(f'The given color palette {
                        palette} has to be either a Callable, list, str, or None.')

    if isinstance(palette, str):
        try:
            import seaborn as sns
        except ImportError as e:
            raise Exception(
                "Named palettes can only be used if seaborn is installed in your python environment. Run `pip install seaborn` to install it.").with_traceback(e.__traceback__)
        # Converting to SVG rgb format
        cmap = sns.color_palette(palette=palette, as_cmap=True)
        # Seaborn's color maps map from [0,1), 1 exclusive.
        # This means that values will wrap around completely if they get to 1.
        # Multiplying the position with 0.9999999 was the quickest fix.

        def palette_function(pos):
            pos = 1 - pos
            pos *= 0.99999999

            return f'#{str(Color(cmap(pos)[0], cmap(pos)[1], cmap(pos)[2]))}'
        _color_palette = palette_function
        return

    _color_palette = palette


def _standard_color_palette(position: float):
    return f'#{str(Color.from_oklch((1-position)**1, -.25*((1-position-.5)**2)+.15, (1-position)*240-140))}'
    # return f'#{str(Color.from_hsl((1-position)**.9*240-180, min(.9,.85*(1-position)**.95+.2), .85*(1-position)**.95+.07))}'
    # return f'#{str(Color.from_hsl((1-position)*180, 1, .6))}'


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
    '''
    Takes in a float between 0 and 1 and returns a color according to the currently selected palette.

    Args:
        position (float): The position on the gradient between 0 and 1.

    Returns:
        str: A color value in HTML Hex notation. E.g. `'#FF0000'`
    '''
    # Standard palette given by the library, 360 different hues but gets
    # crammed / hard to distinguish after 10
    if _color_palette is None:
        return _standard_color_palette(position)

    # Use user-defined palettes
    if isinstance(_color_palette, list):
        return _list_color(_color_palette, position)

    if isinstance(_color_palette, Callable):
        return _color_palette(position)
    import warnings
    warnings.warn(
        "The user-defined palette is not valid. Resorting to default.")
    return _standard_color_palette(position)


def get_contrasting_font_color(color_hex: str):
    color = Color.from_hex(color_hex)
    lightness = color.oklab["l"]
    if lightness < .7:
        return "ffffffff"
    else:
        return "000000ff"


def get_warning_font_color():
    return _standard_color_palette(1)


# Text Palettes
_text_palette: Callable[[float], str] | str | None = None

_text_palettes = {
    # ASCII
    'contrast': list('/░▒▓█-'),
    'numeric': list('/0123456789-'),

    # Emoji
    'magma': list('✖⬛🟦🟪🟥🟧🟨⬜➖'),
    'smiley': list('🫥😭😢🥺😟😐😯😲😃🤩➖'),
    'vehicle': list('⚰🥾🛹🚲🚌🚗🚄🛬🚀➖'),
    'heat': list('🕳🧊⛄💧🌳🔥🌋🌞➖'),
    'moon': list('💤🌑🌒🌓🌔🌕➖')
}
'''
The set of all named text palettes. The first character always stands
for missing data (i.e. no evals left from existing data). The second stands for missing data on the right side. The remaining characters are evenly large bins from 0% to 100%.

It contains the following ASCII palettes:
- contrast: `/░▒▓█-`
- numeric: `/0123456789-`

It contains the following Emoji palettes:
- magma: `✖⬛🟦🟪🟥🟧🟨⬜➖`
- smiley: `🫥😭😢🥺😟😐😯😲😃🤩➖`
- vehicle: `⚰🥾🛹🚲🚌🚗🚄🛬🚀➖`
- heat: `🕳🧊⛄💧🌳🔥🌋🌞➖`
- moon: `💤🌑🌒🌓🌔🌕➖`
'''

_standard_text_palette = _text_palettes['contrast']


def set_text_palette(palette: str | list | Callable[[float | None], str] | None = None):
    '''
    Sets the global palette for displaying percentages using text.

    Args:
        palette (str | list | Callable[[float | None], str] | None, optional): The palette you want to choose. When given a string, it chooses from the named palettes. Custom palettes can be given as a character list or a function. For character lists, the first value corresponds to 'no data'. Functions can have arbitrary functionality, but may lead to unexpected behavior. Defaults to None.

    Raises:
        TypeError: Is raised if the palette is neither str, list, None, or function.
        ValueError: Is raised if the palette is invalid.
    '''
    global _text_palette
    if not isinstance(palette, (Callable, list, str, type(None))):
        raise TypeError(f'The given color palette {
                        palette} has to be either a str, list, Callable, or None.')

    if palette is None:
        _text_palette = _standard_text_palette
        return

    # Choosing a named palette
    if isinstance(palette, str):
        if not palette in _text_palettes.keys():
            raise ValueError(f'The palette {palette} does not exist.')
        _text_palette = _text_palettes[palette]
        return

    # Sanity checks for custom list palettes
    if isinstance(palette, list):
        for char in palette:
            if not isinstance(char, str):
                raise ValueError(f'The palette {palette} contains the object {
                                 char}, which is not a 1-character string.')
            if len(char) != 1:
                raise ValueError(f'The palette {palette} contains a string {
                                 char} of length unequal to 1.')

    _text_palette = palette


def get_text_continuous(position: float | None) -> str:
    if isinstance(_text_palette, Callable):
        return _text_palette(position)

    # List palettes
    palette = _text_palette
    if palette is None:
        palette = _standard_text_palette

    if position is None:
        return palette[0]
    if position == -1:
        return palette[-1]
    index = math.floor(position*(len(palette)-2))
    if index >= len(palette) - 2:
        return palette[-2]
    return palette[index+1]
