from __future__ import annotations
import random
import time
from typing import Callable
from .display_components import init_display
from .util import apply_binning, loss_dict_to_list, apply_sliding_window
from .style import get_color_continuous, get_contrasting_font_color, get_warning_font_color, get_text_continuous
from .loss_data import LossData

UTF_PROGRESS_BAR_LENGTH = 20
FRAMES_PER_SECOND = 30


class LossProgressBar:
    last_gradient_id_start: int = 0
    '''
    Every gradient id is a random number in an interval of width 65536 (2**16). This variable identifies the start of this interval. With every gradient id drawn, it is increased by 65536. The effect is that on every system with 4 byte integers, you can draw 65536 gradient ids before looping around and risking a collision. Systems with 8 byte integers may never do so.
    '''

    def __init__(
        self,
        iterations: int,
        relative_window_size: float = 0.05,
        scaling_function: Callable[[float], float] = lambda x: x,
        display_type: str | None = None,
    ):
        self.iterations = iterations
        '''
        The number of iterations one wants to train for. E.g. entire epochs or just mini-batches.
        '''
        # Iterable setup
        self._iterator = iter(range(self.iterations))
        self._epoch: int = None
        '''
        The current epoch.
        '''
        # Storage
        self._losses = {'train_step': LossData('train_step')}
        '''
        The losses that are generated in each training, or validation step. Multiple validations are possible at once, and validations can be done in irregular intervals. Therefore, the validation losses are stored in a dict of LossData objects. Each lower dict uses the epoch as key and the loss at that time as value.
        '''

        # Sliding window
        self.window_size = round(self.iterations * relative_window_size)

        # Value scaling
        self.scaling_function = scaling_function

        # Setting up Display
        self._display_type = display_type
        self._display_components = init_display(display_type)
        if self._display_components is None:
            self._display_type = "text"
        self._display_content = None
        self._last_draw: float = 0

        # warning management
        self._warnings = dict()

    @classmethod
    def get_gradient_id(cls) -> str:
        '''
        Returns a new, unique gradient id, with some randomness to avoid problems with parallel executions.
        '''
        gradient_id = cls.last_gradient_id_start + random.randint(1, 65536)
        cls.last_gradient_id_start += 65536

        return f'''{cls.__module__}{gradient_id}'''

    @staticmethod
    def run(iterations: int,
            train_step: Callable[[], float] | None = None,
            val_interval: int = 1,
            relative_window_size: float = 0.05,
            scaling_function: Callable[[float], float] = lambda x: x,
            display_type: str = None,
            **val_steps: Callable[[], float],
            ):

        loss_progress_bar = LossProgressBar(
            iterations=iterations,
            relative_window_size=relative_window_size,
            scaling_function=scaling_function,
            display_type=display_type)

        for epoch, update in loss_progress_bar:
            if train_step is not None:
                train_loss = train_step()
                update(train_loss)
            # Creating a dict to pass keyword arguments
            val_losses = dict()
            if epoch % val_interval == val_interval - 1:
                for val_step, func in val_steps.items():
                    val_losses[val_step] = func()
                update(**val_losses)

    def __iter__(self):
        return self

    def __next__(self):
        self._epoch = next(self._iterator)
        # Returning the current epoch, as well as an update method
        return self._epoch, lambda train_loss = None, **val_losses: self.update(self._epoch, train_loss, **val_losses)

    def _warn(self, warning: str):
        if self._warnings.get(warning) is not None:
            self._warnings[warning] += 1
        else:
            self._warnings[warning] = 1

    def draw_warnings_html(self):
        ret = '<div>'
        for warning, count in self._warnings.items():
            ret += f'<span style="color:{get_warning_font_color(
            )};font-weight:bold">Warning:</span> {warning}: {count}x\n<br>\n'
        return ret + '</div>'

    def draw_warnings_utf8(self):
        ret = ''
        for warning, count in self._warnings.items():
            ret += f'Warning: {warning}: {count}x\n'
        return ret

    def update(self,
               iteration: int,
               train_loss: float | None = None,
               **val_losses: float
               ):
        if train_loss is not None:
            self._losses['train_step'][iteration] = train_loss

        for name, loss in val_losses.items():
            if self._losses.get(name) is None:
                self._losses[name] = LossData(name)
            if val_losses[name] is not None:
                self._losses[name][iteration] = loss
        # Only draw at most at given Framerate
        current_time = time.time()
        if current_time - self._last_draw >= 1/FRAMES_PER_SECOND or iteration == self.iterations-1:
            self.draw(iteration)
            self._last_draw = current_time

    def draw(self, epoch: int):
        # Training and validation step should share an overall minimum and maximum
        overall_min = float('inf')
        overall_max = float('-inf')

        for other_loss_name, losses in self._losses.items():
            if self._losses[other_loss_name].is_empty:
                continue
            overall_min = min(
                overall_min, *(losses.values()))
            overall_max = max(
                overall_max, *(losses.values()))
        if self._display_type == "text":
            self.draw_utf_8(epoch, overall_min, overall_max)
        else:
            self.draw_svg(epoch, overall_min, overall_max)

    def draw_utf_8(self, epoch: int, overall_min: float, overall_max: float):
        content = ''
        for other_loss_name, losses in self._losses.items():
            if losses.is_empty:
                continue
            content += self.create_utf8_loading_bar(
                epoch,
                loss_dict_to_list(losses, epoch),
                visual_min_loss=overall_min,
                visual_max_loss=overall_max,
                text_min_loss=min(losses.values()),
                text_max_loss=max(losses.values()),
                name=other_loss_name
            )

        if self._display_components is None:
            # If we already have something, we want to clean the previous lines
            if self._display_content is not None:
                # We assume that the display content is text
                lines = self._display_content.count("\n") + 1
                # Moving up all the lines and clearing the line
                for _ in range(lines):
                    # Move the cursor up and clear the line
                    print("\33[1A", end="\x1b[2K")
            # printing the content
            self._display_content = content + self.draw_warnings_utf8()
            print(self._display_content)
        else:
            content = content.replace("\n", "<br>").replace(" ", "&nbsp;")
            style = '''
            <style>
                * {
                    font-family: monospace;
                }
            </style>
            '''
            content = style + content + self.draw_warnings_html()
            if self._display_content is None:
                self._display_content = self._display_components["HTML"](
                    value=content)
                self._display_components["display"](self._display_content)
            else:
                self._display_content.value = content

    def draw_svg(self, epoch, overall_min, overall_max):
        progress_bar_html = '''
            <style>
            .cell-output-ipywidget-background {
            background-color: transparent !important;
            }
            .jp-OutputArea-output {
            background-color: transparent;
            }  
            </style>
            '''

        for other_loss_name, losses in self._losses.items():
            if losses.is_empty:
                continue
            val_progress_bar_svg = self.create_svg_progress_bar(
                epoch,
                loss_dict_to_list(losses, epoch),
                visual_min_loss=overall_min,
                visual_max_loss=overall_max,
                text_min_loss=min(losses.values()),
                text_max_loss=max(losses.values()),
                name=other_loss_name
            )
            progress_bar_html += f"<br>{val_progress_bar_svg}"

        # Adding warnings
        progress_bar_html += self.draw_warnings_html()
        # Initializing or updating the displayed output

        if self._display_content is None:
            # Init
            self._display_content = self._display_components["HTML"](
                progress_bar_html)
            self._display_components["display"](self._display_content)
        else:
            # Update: ipywidgets automatically rerenders the bars
            self._display_content.value = progress_bar_html

    def create_utf8_loading_bar(self,
                                epoch: int,
                                losses: list[float],
                                visual_min_loss: float,
                                visual_max_loss: float,
                                text_min_loss: float,
                                text_max_loss: float,
                                name: str
                                ):
        progress = epoch / (self.iterations-1)

        if visual_max_loss != visual_min_loss:
            normalized_losses = [
                None if l is None else (l - visual_min_loss) / (visual_max_loss - visual_min_loss) for l in losses]
        else:
            normalized_losses = [
                None if l is None else 0 for l in losses]
        binned_losses = apply_binning(
            normalized_losses, self.iterations/UTF_PROGRESS_BAR_LENGTH)
        content = ""
        for binned_loss in binned_losses:
            if binned_loss is None:
                content += get_text_continuous(None)
                continue
            content += get_text_continuous(1-binned_loss)

        content += get_text_continuous(-1) * \
            (UTF_PROGRESS_BAR_LENGTH-len(content))
        content = "[" + content + "]"
        content += "  "

        # Constructing the text elements
        name_string = f'{name[:10]:<10}'.rjust(10)
        progress_string = f'{str(round(progress * 100))}%'.rjust(4)
        loss_string = f'Loss: {losses[epoch]:.3g}'.ljust(15)
        min_loss_string = f'Min: {text_min_loss:.3g}'.ljust(14)
        max_loss_string = f'Max: {text_max_loss:.3g}'.ljust(14)
        content += name_string
        content += ': '
        content += progress_string
        content += ' |'
        content += loss_string
        content += ' |'
        content += get_text_continuous(
            0) if visual_max_loss == text_max_loss else ' '
        content += max_loss_string
        content += get_text_continuous(
            0) if visual_max_loss == text_max_loss else ' '
        content += '|'
        content += get_text_continuous(
            1) if visual_min_loss == text_min_loss else ' '
        content += min_loss_string
        content += get_text_continuous(
            1) if visual_min_loss == text_min_loss else ' '
        return content + "\n"

    def create_svg_progress_bar(
        self,
        epoch: int,
        losses: list[float],
        visual_min_loss: float,
        visual_max_loss: float,
        text_min_loss: float,
        text_max_loss: float,
        name: str,
    ):
        progress = epoch / (self.iterations - 1)

        averaged_losses = apply_sliding_window(losses, self.window_size)
        if visual_max_loss != visual_min_loss:
            normalized_losses = [
                None if l is None else (l - visual_min_loss) / (visual_max_loss - visual_min_loss) for l in averaged_losses]
        else:
            normalized_losses = [
                None if l is None else 0 for l in averaged_losses]

        def gradient_stops(normalized_losses):
            ret = ""
            for i, loss in enumerate(normalized_losses):
                if loss is None:
                    ret += f"""<stop offset="{i * 100 / len(
                        normalized_losses)}%" style="stop-color:hsl(0,100%,0%);stop-opacity:1"/>\n"""
                    continue
                ret += f"""<stop offset="{i * 100 / len(normalized_losses)}%" style="stop-color:{
                    get_color_continuous(loss)};stop-opacity:1"/>\n"""
            return ret
        gradient_id = self.get_gradient_id()

        svg_string = '<svg width="900" height="40" xmlns="http://www.w3.org/2000/svg">\n'
        # Defining the gradient
        svg_string += f'''
            <defs>
                <linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">
                    {gradient_stops(normalized_losses)}
                </linearGradient>
            </defs>
        '''
        # Outer rectangle
        svg_string += '''<rect x="1" y="1" width="380" height="20" fill="none" stroke="grey" stroke-width="2" rx="2" ry="2"/>'''
        # Inner rectangle with gradient
        svg_string += f'''<rect x="2" y="2" width="{progress * 378}
            " height="18" fill="url(#{gradient_id})" rx="1" ry="1"/>'''

        # Text configuration
        text_x_position = 400
        text_y_position = 15

        # Helper function to create background text rectangle and text element
        def create_text_with_background(text, x, y, bg_color, width, font_color="black"):
            bg_rect = f'<rect x="{x}" y="{y - 12}" width="{width*7 + 8}" height="16" fill="{
                bg_color}" rx="2" ry="2" margin="2" stroke-width="1" stroke="#cccccc"/>'
            text_elem = f'<text x="{
                x+3}" y="{y}" fill="{font_color}" font-size="12" font-family="monospace">{text}</text>'
            return bg_rect + text_elem

        # Constructing the text elements
        type_string = f'{name[:10]:<10}'.rjust(10).replace(" ", "&nbsp;") + ":"
        progress_string = f'{str(round(progress * 100))}%'
        loss_string = f'Loss: {losses[epoch]:.3g}'
        min_loss_string = f'Min: {text_min_loss:.3g}'
        max_loss_string = f'Max: {text_max_loss:.3g}'

        # Checking if min and max should be highlighted
        max_bg_color = get_color_continuous(
            1) if text_max_loss == visual_max_loss else "#FFFFFFFF"
        min_bg_color = get_color_continuous(
            0) if text_min_loss == visual_min_loss else "#FFFFFFFF"
        max_font_color = '#' + get_contrasting_font_color(max_bg_color)
        min_font_color = '#' + get_contrasting_font_color(min_bg_color)

        # Adding text with background rectangles
        svg_string += create_text_with_background(
            type_string, text_x_position, text_y_position, "white", 11)
        text_x_position += 11 * 7 + 10
        svg_string += create_text_with_background(
            progress_string, text_x_position, text_y_position, "white", 4)
        text_x_position += 4 * 7 + 10
        svg_string += create_text_with_background(
            loss_string, text_x_position, text_y_position, "white", 15)
        text_x_position += 15 * 7 + 10
        svg_string += create_text_with_background(
            max_loss_string, text_x_position, text_y_position, max_bg_color, 14, max_font_color)
        text_x_position += 14 * 7 + 10
        svg_string += create_text_with_background(
            min_loss_string, text_x_position, text_y_position, min_bg_color, 14, min_font_color)

        svg_string += '</svg>'
        return svg_string
