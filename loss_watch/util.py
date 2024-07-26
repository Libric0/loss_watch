'''
A collection of helpful functions that may be used for more than one kind of
loss plot.
'''
import statistics
import math


def loss_dict_to_list(
        dict_val_losses: dict[int, float],
        epoch: int
) -> list[float | None]:
    '''
    Takes a loss dict, containing epochs as keys and losses at that time as
    values, and the current epoch; and returns a list containing the losses.
    Gaps are filled with the last known loss. If the first epoch is not
    evaluated, this gap will be filled with `None`s.

    ## Example
    ```python
    val_losses = {3:1.3, 5:0.8}
    print(loss_dict_to_lost(val_losses))
    >>> [None, None, None, 1.3, 1.3, 0.8]
    ```
    '''
    ret = []
    # Bookkeeping for "interpolation"
    last_evaluated_epoch = 0
    # Setting last_eval_loss to none has the effect that everything before the
    # first evaluated epoch will have eval loss None. This can then be rendered
    # differently, e.g. in black.
    last_eval_loss = None
    # Adding the actual evaluations to the list
    for evaluated_epoch in sorted(dict_val_losses.keys()):
        # We only add the evaluations once we know how many we have to add
        # (i.e. once we know when the next epoch was).
        ret += [last_eval_loss] * (evaluated_epoch - last_evaluated_epoch)
        last_evaluated_epoch = evaluated_epoch
        last_eval_loss = dict_val_losses[evaluated_epoch]
    # Since we dont add an epoch's own loss within the loop, the last
    # validation loss is ignored. Adding it here, until the current epoch
    ret += [last_eval_loss] * (epoch - last_evaluated_epoch + 1)
    return ret


def apply_sliding_window(values: list[float | None], window_size: int):
    '''
    Assumption: values can start with Nones, but cannot have None after the
    first value. E.g. `loss_dict_to_list` was applied first.
    '''
    if window_size == 0:
        return values
    index_without_none = None
    for i, v in enumerate(values):
        if v is not None:
            index_without_none = i
            break
    # No values -> no sliding window
    if index_without_none is None:
        return values

    values_without_none = values[index_without_none:]

    return [None]*(index_without_none) + [
        statistics.mean(values_without_none[max(0, i-window_size): i+1])
        for i, _ in enumerate(values_without_none)]


def apply_binning(values: list[float | None], bin_size: float):
    '''
    Applies continuous binning to a list of values. Continuous, because
    `bin_size` may be non-integer. Values between integer borders are then
    partly added to the first bin, and partly to the last.

    Assumption: values can start with Nones, but cannot have None after the
    first value. E.g. `loss_dict_to_list` was applied first.
    '''
    bins = []
    # If we have bin size 2.5 and would like to bin 6 values, we need ceil(6/2.5) = 3 bins.
    for i in range(math.ceil(len(values)/bin_size)):
        # Current bin size might change depending on whether a bin starts with Nones
        # Marking the edges of the bins
        float_bin_start = min(i*bin_size, len(values)-1)
        int_bin_start = math.ceil(float_bin_start)
        lowest_index = math.floor(float_bin_start)
        lowest_value = values[lowest_index]

        float_bin_end = min(len(values), (i+1)*bin_size) - 1
        int_bin_end = math.floor(float_bin_end)
        highest_index = math.ceil(float_bin_end)
        highest_value = values[highest_index]

        # Marking current bin size. +1 is needed because we always
        # subtract one from float_bin_end so that the highest index
        # stays the same for integer indices, no matter floor or ceil.
        current_bin_size = float_bin_end-float_bin_start + 1

        if highest_value is None:
            # If the highest value a bin contains is None, this is an empty bin so we add None.
            bins.append(None)
            continue
        if float_bin_start == float_bin_end:
            # We are at the last one, no mean needed
            bins.append(values[-1])
            break
        if lowest_value is None:
            # This has to be a bin with a Number, otherwise highest_value would be None too.
            # We just adjust the starting indices
            lowest_index = values.count(None)
            int_bin_start = lowest_index
            float_bin_start = lowest_index
            lowest_value = values[lowest_index]
            current_bin_size = float_bin_end-float_bin_start + 1
            # If it is the last, and even if only partly, we want this to be the whole bin
        if lowest_index == highest_index:
            # Bin of size 1
            bins.append(values[highest_index])
            continue

        # Calculating the bin value for uneven bins
        integer_slice = values[int_bin_start: int_bin_end+1]

        # Getting influences of partly binned objects
        start_value_influence = int_bin_start - float_bin_start
        end_value_influence = float_bin_end - int_bin_end

        # Applying the influence and calculating the mean
        integer_slice.append(highest_value*end_value_influence)
        integer_slice.append(lowest_value*start_value_influence)
        bins.append(sum(integer_slice)/current_bin_size)

    return bins
