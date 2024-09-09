'''
    A class that manages loss data and gives quick access to loss statistics.
'''


class LossData:
    def __init__(self, running_alpha: float = 0.05):
        # Settings
        self.running_alpha = running_alpha

        # Losses
        self.losses = dict()
        '''
        A dictionary that maps the iteration to the loss at that iteration.
        '''

        # Usual Statistics
        self.min: float = float('-inf')
        self.min_iteration = None
        self.max: float = float('inf')
        self.max_iteration = None
        self.sum: float = 0.0
        self.latest_iteration: int = 0
        '''
        The latest iteration in which a loss value was added to this data object.
        '''

        # Running Statistics
        self.running_mean: float | None = None
        self.running_derivative: float | None = None

    @property
    def mean(self) -> float:
        return self.sum/len(self.losses)

    # Dictionary Methods
    def __getitem__(self, key):
        return self.losses[key]

    def __setitem__(self, key, value):
        if self.losses.get(key) is not None:
            # Updating an existing loss.
            pass
        elif key < self.latest_iteration:
            # Updating something in between
            # Note that the = case is already covered by 1st condition.
            pass
        else:
            # Adding a brand new loss
            pass
