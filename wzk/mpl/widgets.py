from matplotlib import widgets
from wzk.math2 import modulo


# Widgets
def create_button(fig, axes, listener_fun, name='button'):
    b_ax = fig.axes(axes)
    b = widgets.Button(b_ax, name)
    b.on_clicked(listener_fun)
    return b


def create_key_slider(ax, callback,
                      label, valfmt, valmin, valmax, valinit, valstep,
                      fast_step=10):
    slider = widgets.Slider(ax=ax,
                            label=label, valfmt=valfmt,
                            valmin=valmin, valmax=valmax, valinit=valinit, valstep=valstep)

    def cb_key(event):
        val = slider.val
        if event.key == 'right':
            slider.set_val(modulo(val+valstep, low=valmin, high=valmax))

        if event.key == 'left':
            slider.set_val(modulo(val-valstep, low=valmin, high=valmax))

        if event.key == 'up':
            slider.set_val(modulo(val+fast_step*valstep, low=valmin, high=valmax))

        if event.key == 'down':
            slider.set_val(modulo(val-fast_step*valstep, low=valmin, high=valmax))

    slider.on_changed(callback)
    keyboard = ax.get_figure().canvas.mpl_connect('key_press_event', cb_key)
    return slider, keyboard
