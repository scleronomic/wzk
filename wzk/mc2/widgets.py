import numpy as np

from pynput import keyboard
import threading


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

        self.listener = None
        self.__thread = None
        self.start_listener()

    def start_listener(self):
        self.__thread = threading.Thread(target=self.__start_listener)
        self.__thread.daemon = True
        self.__thread.start()

    def __start_listener(self):
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, k):
        # todo provide here a dict, to handle all keys in a single thread/function
        #   return the dict and extend it if necessary. I guess for this you have to
        #   first join the existing thread and then start a new one.
        #   two parts one is just the key listener with dict, the other are dedicated keylisteners left right up down wasd etc
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

        # self.back_k
        self.factor = int((ma - mi) / 20)
        self.left = KeyListener(key="left", callback=lambda: self._update(step=-self.step))
        self.right = KeyListener(key="right", callback=lambda: self._update(step=+self.step))
        # self.up = KeyListener(key="up", callback=lambda: self._update(step=-self.step*self.factor))
        # self.down = KeyListener(key="bottom", callback=lambda: self._update(step=+self.step*self.factor))

        # self.left.listener.start()  # start to listen on a separate thread
        # self.right.listener.start()  # start to listen on a separate thread
        # self.up.listener.start()  # start to listen on a separate thread
        # self.down.listener.start()  # start to listen on a separate thread

        # listener.join()  # remove if main thread is polling self.keys

    def _update(self, step):
        v2 = self.value + step
        v2 = np.clip(v2, a_min=self.min, a_max=self.max)
        self.value = v2
        self.callback(v2)


def try_KeySlider():

    def fun(i):
        print("fun")
        print(i)

    return KeySlider(callback=fun, step=1, mi=0, ma=1000)


if __name__ == "__main__":
    try_KeySlider()

