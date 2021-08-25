import copy


class CopyableObject(object):
    __slots__ = ()

    def copy(self):
        return copy.copy(self)
