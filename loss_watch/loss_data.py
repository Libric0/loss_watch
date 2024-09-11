'''
    A class that manages loss data and gives quick access to loss statistics.
'''


class LossData:
    def __init__(self, name: str, running_alpha: float = 0.05):
        '''
        A class that manages loss data and gives quick access to loss statistics

        Args:
            name (str): The name of this loss. Used to give more information in warnings.
            running_alpha (float, optional): The influence factor of new loss data on running statistics. A higher value means that new losses have a higher influence and the current estimates decay quickly. Defaults to 0.05.
        '''
        # Attributes
        self.name = name

        # Settings
        self.running_alpha = running_alpha

        # Losses
        self._losses = dict()
        '''
        A dictionary that maps the iteration to the loss at that iteration.
        '''

        # Usual Statistics
        self._minimum: float = float('-inf')
        self._minimum_iteration = None
        self._maximum: float = float('inf')
        self._maximum_iteration = None
        self._sum: float = 0.0
        self._latest_iteration: int = -1
        '''
        The latest iteration in which a loss value was added to this data object.
        '''
        self.num_finished_updates: int = None

        # Running Statistics
        self._running_mean: float | None = None
        self._running_derivative: float | None = None

    # Properties
    # Functional Properties
    @property
    def is_empty(self):
        return not self._losses
    # Loss Statistics
    # Simple, no calculation needed. Should be read-only variables
    # maintained by the object

    @property
    def minimum(self):
        '''
        The lowest loss recorded by this class.

        Returns:
            float: The loss.
        '''
        return self._minimum

    @property
    def minimum_iteration(self):
        '''
        The earliest iteration of the lowest loss recorded by this class.

        Returns:
            int: The iteration.
        '''
        return self._minimum_iteration

    @property
    def maximum(self):
        '''
        The highest loss recorded by this class.

        Returns:
            float: The loss.
        '''
        return self._maximum

    @property
    def maximum_iteration(self):
        '''
        The earliest iteration of the highest loss recorded by this class.

        Returns:
            int: The iteration.
        '''
        return self._maximum_iteration

    @property
    def sum(self):
        '''
        The sum of all losses recorded so far.

        Returns:
            float: The sum.
        '''
        return self._sum

    @property
    def latest_iteration(self):
        '''
        The latest iteration a loss was recorded for.

        Returns:
            int: The iteration.
        '''
        return self._latest_iteration
    # Properties that need to be calculated on the fly

    @property
    def mean(self) -> float:
        return self._sum/len(self._losses)

    # Internal Calculations
    def _update_minimum(self, iteration, loss):
        if loss < self._minimum:
            self._minimum = loss
            self._minimum_iteration = iteration

    def _update_maximum(self, iteration, loss):
        if loss > self._maximum:
            self._maximum = loss
            self._maximum_iteration = iteration

    def _update_running_statistics(self, iteration, loss):
        if self._running_mean is None:
            # Initializing Running Mean as the loss
            self._running_mean = loss
            return

        if self._running_mean is not None and self._running_derivative is None:
            # Initializing running derivative as the derivative estimate
            difference = loss - self._running_mean
            iter_diff = iteration - self._latest_iteration
            deriv_estimate = difference/iter_diff
            self._running_derivative = deriv_estimate
            # Updating running mean
            self._running_mean = self.running_alpha*loss + \
                (1-self.running_alpha)*self._running_mean
            return

        # Updating running derivative
        difference = loss - self._running_mean
        iter_diff = iteration - self._latest_iteration
        deriv_estimate = difference/iter_diff
        self._running_derivative = self.running_alpha*deriv_estimate + \
            (1-self.running_alpha)*self._running_derivative

        # Updating running mean
        self._running_mean = self.running_alpha*loss + \
            (1-self.running_alpha)*self._running_mean
        pass
    # Dictionary Methods

    def __getitem__(self, key):
        return self._losses[key]

    def __setitem__(self, key: int, value: float | None):
        if value is None:
            # We actually did not get any value
            return
        if key == self._latest_iteration \
                and self._losses.get(key) is not None \
                and value != self._losses.get(self._latest_iteration):
            print(f'Warning: {self.name} was updated multiple times with different values within the same epoch. Statistics may be corrupted.')  # noqa
        if key < self._latest_iteration:
            print(
                'Warning: Updating the loss at an earlier than latest iteration. Statistics may be corrupted.')
        # All statistics are updated before the new loss is added to self._losses
        self._update_minimum(key, value)
        self._update_maximum(key, value)
        if key > self._latest_iteration:
            self._update_running_statistics(key, value)

        self._sum += value
        self._latest_iteration = max(self._latest_iteration, key)

        self._losses[key] = value

    def __delitem__(self, key):
        del self._losses[key]

    def __contains__(self, key):
        return key in self._losses

    def keys(self):
        return self._losses.keys()

    def values(self):
        return self._losses.values()

    def items(self):
        return self._losses.items()
