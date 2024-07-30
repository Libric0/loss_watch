# Loss Watch
Loss-watch is a library that allows you to watch your training / validation losses in a neat and orderly way, both for your jupyter notebooks and your console. 

I created it because I was tired of endless blocks of printed losses in my console / notebooks. Loss watch, instead, gives you a progress bar with colors (or contrasts in the terminal) indicating your model's progress.

## Installation
```bash
pip install loss-watch
```

## Usage
Similarly to `tqdm`, loss-watch plots act as the iterable you loop through while training your model. The simplest way of using it is as follows

```python
from loss_watch import LossProgressBar

epochs = 100
for epoch, update in LossProgressBar(epochs):
    # Perform your models training step and retrieve a float `loss`
    train_loss = train_step()
    update(train_loss)
```

It does not really matter how you get your training loss here, any float works. This will give you a plot that looks something like this:
![img](images/train_loss_plot_1.png)

As you can see, your highest loss is displayed in red, and the lowest in a light cyan. 

### Validation
Once in a while, you would probably like to validate your model on one or multiple validation sets. Your Loss progress bar can handle as many as you like. And in contrast to your training loss, you can validate in any interval you like! Simply pass a named float to the `update` function, and it will generate another progress bar, that corresponds to this name, for you.

If you extend our example to the following:
```python
for epoch, update in LossProgressBar(epochs):
    # Perform your models training step and retrieve a float `loss`
    train_loss = train_step()
    if epoch % 10 == 9:
        # A cheap validation
        val_loss1 = val_step()
        update(val_step1=val_loss1)
    if epoch % 25 == 24:
        # A more expensive validation
        val_loss2 = val_step2()
        update(val_step2=val_loss2)
    update(train_loss)
```
As you can see, you can call update multiple times. You can also not call update at all for any of your losses *including the train loss*. Missing values will be interpolated.

The resulting plot will look something like this:
![img](images/train_and_val_loss_plot_1.png)

The black parts indicate that there was no validation data yet. You can also see that the min loss and max loss of the training step are marked in red and cyan. This is because all plots will share the colors for the minimum and maximum loss, making it easier for you to compare them. The marked losses correspond to the overall minimum and maximum here, respectively. 

### Quick Mode
While the above examples always illustrated the training and validation steps as functions returning a float, you are allowed to obtain these values however you like. However, if you already packaged your training/validation logic into functions, you can plot your losses in a one-liner:
```python
LossProgressBar.run(epochs=epochs, train_step=train_step, val_step1=val_step, val_step2=val_step2)
```

The above code will run each training and validation step once per epoch.

> **Hint:** 
> If you don't want to run every validation in every epoch, you will have to define a stateful variable e.g. `epoch` yourself and check within the validation step whether it is time to validate.

### Palettes
If you don't like the standard palette, you can optionally use seaborn colormaps as well. Make sure seaborn is installed, and before your training loop write for example:
```
from loss_watch import set_palette
set_palette("Spectral")
```

This will select the `Spectral` colormap and use this to display your plot. Your losses, then, will look something like this:
![img](images/spectral_palette.png)

Of course, this feature does not work in the terminal.
