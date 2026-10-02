from tensorboardX import SummaryWriter


class Recorder:
    _active_writers = set()

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.__class__._active_writers.add(self.writer)

    def record(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer in self._active_writers:
            self.writer.flush()
            self.writer.close()
            self._active_writers.remove(self.writer)

    def __del__(self):
        self.close()
