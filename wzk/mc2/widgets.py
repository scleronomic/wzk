import numpy as np

from pynput import keyboard


def wrapper_key(key):
    try:
        k = key.char  # single-char keys
    except AttributeError:
        k = key.name  # other keys
    return k


class KeyListener:
    def __init__(self, key, callback):
        self.key = key
        self.callback = callback
        self.listener = keyboard.Listener(on_press=self.on_press)

    def on_press(self, k):
        k = wrapper_key(k)
        if k == self.key:
            self.callback()


class KeySlider:

    def __init__(self, callback, step, mi, ma):
        self.value = 0

        self.callback = callback
        self.step = step
        self.min = mi
        self.max = ma

        self.back_k
        self.factor = 10
        listener = KeyListener(key="left", callback=lambda: self._update(step=-self.step))
        listener = KeyListener(key="right", callback=lambda: self._update(step=+self.step))
        listener = KeyListener(key="up", callback=lambda: self._update(step=-self.step*self.factor))
        listener = KeyListener(key="bottom", callback=lambda: self._update(step=+self.step*self.factor))

        # listener.start()  # start to listen on a separate thread
        # listener.join()  # remove if main thread is polling self.keys

    def _update(self, step):
        v2 = self.value + step
        v2 = np.clip(v2, a_min=self.min, a_max=self.max)
        self.value = v2
        self.callback(v2)

    def _backward(self, key):
        k = wrapper_key(key)
        if k == "left":
            self._update(-self.step)

    def _forward(self):
        pass

    def _backward10(self):
        pass

    def _forward10(self):
        pass


def add_key_slider_widget(vis, callback, step=1, mi=0, ma=1):

    def __on_key(s):
        v2 = r.GetValue() + s
        v2 = np.clip(v2, a_min=mi, a_max=ma)
        r.SetValue(v2)
        callback(v2)
        pl.render()

    def on_left():
        __on_key(s=-step)

    def on_right():
        __on_key(s=+step)

    def on_down():
        __on_key(s=-step*100)

    def on_up():
        __on_key(s=+step*100)

    pl.add_key_event(key="Left", callback=on_left)
    pl.add_key_event(key="Right", callback=on_right)
    pl.add_key_event(key="Down", callback=on_down)
    pl.add_key_event(key="Up", callback=on_up)