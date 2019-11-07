import queue
from enum import Enum

import carla


class Side(Enum):
    Top = 1
    Bottom = 2
    Front = 3
    Back = 4
    Right = 5
    Left = 6

    def rotation(self):
        if self is Side.Top:
            return carla.Rotation(90, 0, 0)
        elif self is Side.Bottom:
            return carla.Rotation(-90, 0, 0)
        elif self is Side.Front:
            return carla.Rotation(0, 0, 0)
        elif self is Side.Back:
            return carla.Rotation(0, 180, 0)
        elif self is Side.Right:
            return carla.Rotation(0, 90, 0)
        elif self is Side.Left:
            return carla.Rotation(0, -90, 0)
        else:
            raise ValueError(f"No known side for {self}")


class SideMap:
    def __init__(self):
        self.sides = {}

    def __setitem__(self, key: Side, value):
        q = queue.Queue()
        value.listen(q.put)
        self.sides[key] = (value, q)
        return q

    def camera(self, key: Side):
        return self.sides[key][0]

    def queue(self, key: Side) -> queue.Queue:
        return self.sides[key][1]

    def has(self):
        return all(not x[1].empty() for x in self.sides.values())

    def pop(self, frame, timeout=2.0, fn=lambda x: x):
        data = {}
        for k, v in self.sides.items():
            while True:
                d = v[1].get(timeout=timeout)
                if d.frame == frame:
                    data[k] = fn(d)
                    break

        return data
