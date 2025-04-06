import psutil
import torch
import collections
import datetime
import time
import torch
import torch.distributed as dist

def reduce_dict(input_dict, average=True):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = list(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys])
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt if fmt is not None else "{median:.4f} ({global_avg:.4f})" 

    def update(self, value, n=1):
        self.deque.append(value)
        self.total += value * n
        self.count += n

    @property
    def median(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.median().item() if len(d) > 0 else 0.0

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if len(d) > 0 else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    def __str__(self):
        if not self.deque:
            return "No values"
        return f"Median: {self.median:.4f}, Avg: {self.avg:.4f}, Global Avg: {self.global_avg:.4f}"





class MetricLogger:
   
    def __init__(self, delimiter="\t"):
        self.meters = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join([
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")

def get_ram():
    mem = psutil.virtual_memory()
    free = mem.available / 1024 ** 3
    total = mem.total / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'RAM: {total - free:.2f}/{total:.2f}GB\t RAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'


def get_vram():
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def print_memory():
    print(get_ram(), get_vram())

def collate_fn(batch):
    return tuple(zip(*batch))


