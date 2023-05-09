import numpy as np

from pynput import keyboard
from wzk import np2


class KeyListener:
    def __init__(self, key2callback: dict = None):
        self.key2callback = key2callback

        self.listener = None
        self.is_listening = False
        self.start_listening()

    @staticmethod
    def wrapper_key(key):
        try:
            k = key.char  # single-char keys
        except AttributeError:
            k = key.name  # other keys
        return k

    def add_callback(self, key2callback: dict = None, start_listening=True):
        self.key2callback.update(key2callback)
        if start_listening:
            self.start_listening()

    def start_listening(self, on_press=True, on_release=False):
        if self.is_listening:
            self.stop_listening()

        if on_press and not on_release:
            self.listener = keyboard.Listener(on_press=self.on_press)
        elif on_release and not on_press:
            self.listener = keyboard.Listener(on_release=self.on_press)
        else:

            raise NotImplementedError

        self.listener.start()
        self.is_listening = True

    def stop_listening(self):
        self.listener.stop()
        self.listener.join()
        self.is_listening = False

    def on_press(self, k):
        k = self.wrapper_key(k)
        if k in self.key2callback:
            self.key2callback[k]()


class KeySlider:

    def __init__(self, callback, step, mi, ma, periodic=False):
        __factor = 20
        self.value = 0

        self.callback = callback
        self.step = step
        self.min = mi
        self.max = ma
        self.periodic = periodic

        self.factor = int((ma - mi) / __factor)

        # Create KeyListener
        key2callback = dict(left=lambda: self._update(step=-self.step),
                            right=lambda: self._update(step=+self.step),
                            up=lambda: self._update(step=-self.step*self.factor),
                            down=lambda: self._update(step=+self.step*self.factor))
        self.listener = KeyListener(key2callback=key2callback)

    def _update(self, step):
        v2 = self.value + step
        if self.periodic:
            v2 = np2.clip_periodic(x=v2, a_min=self.min, a_max=self.max)
        else:
            v2 = np.clip(x=v2, a_min=self.min, a_max=self.max)

        self.value = v2
        self.callback(v2)


def try_KeySlider():

    def fun(i):
        print("fun")
        print(i)

    return KeySlider(callback=fun, step=1, mi=0, ma=100, periodic=True)


def try_KeyListener():
    key2callback = dict(w=lambda: print("ww"),
                        a=lambda: print("aa"),
                        s=lambda: print("ss"),
                        d=lambda: print("dd")
                        )
    kl = KeyListener(key2callback=key2callback)
    return kl


if __name__ == "__main__":
    pass
    # try_KeyListener()
    # try_KeySlider()
