

def init_display(display_type: str | None = None):
    # Default: Jupyter
    if _is_in_ipython_notebook() or display_type == "ipython":
        # If we set the display type to ipython, we want to get any errors
        # That may be raised.
        from IPython.display import display
        from ipywidgets import HTML
        return {"display": display, "HTML": HTML}

    # Resorting to print statements
    return None


def _is_in_ipython_notebook():
    '''
    Checks whether ipython is available.
    '''
    try:
        from IPython.display import display
        from IPython import get_ipython
        from ipywidgets import HTML

        return get_ipython() is not None
    except ImportError:
        return False
