from __future__ import absolute_import
from collections import OrderedDict
from typing import overload, List, Tuple, Union, Dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None, Lambda=0.9, max=1024):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.Lambda = Lambda
        self.max = max

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val if val < self.max else self.max
        self.sum += val * n
        self.count += n
        self.avg = self.Lambda * self.avg + (1 - self.Lambda) * self.val if self.count != n else self.val

    def __call__(self, global_avg=False):
        return [self.avg if global_avg is False else (self.sum / self.count),
                self.val]

    def conclude(self):
        return self.sum / self.count


class AverageMeters(object):
    def __init__(self, *meters):
        self.meters = OrderedDict([(meter.name, meter) for meter in meters])

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    @overload
    def update(self, values: Union[List, Tuple], n: int=1) -> None: ...

    @overload
    def update(self, values: Dict, n: int=1) -> None: ...

    def update(self, values, n=1):
        if isinstance(values, (list, tuple)):
            assert len(self.meters) == len(values), "Invalid number of values."
            for value, meter in zip(values, self.meters.values()):
                meter.update(value, n)
        elif isinstance(values, dict):
            for name, value in values.items():
                self.meters[name].update(value, n)
        else:
            raise RuntimeError("Invalid type for 'values'. ")

    def __call__(self, global_avg=False):
        info = ", ".join(["{}: {:.4f}/{:.4f}".format(name, *meter(global_avg=global_avg))
                          for name, meter in self.meters.items()])
        return info

    def append(self, meter):
        self.meters[meter.name] = meter
