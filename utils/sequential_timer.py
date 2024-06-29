from time import perf_counter


class SequentialTimer:
    def __init__(self):
        self.times = []

    def time(self, name):
        self.times.append((name, perf_counter()))

    def print(self):
        print(self.__str__())

    def __str__(self):
        first = self.times[0][1]
        last = self.times[-1][1]
        s = ""
        for (n1, t1), (_, t2) in zip(self.times, self.times[1:]):
            if n1 == "END":
                continue
            s += f"{t2 - t1:.4f} | {n1}\n"
        s += f"{last - first:.4f} | ALL\n"
        return s

    def get_json(self):
        res = []
        first = self.times[0][1]
        last = self.times[-1][1]
        for (n1, t1), (_, t2) in zip(self.times, self.times[1:]):
            if n1 == "END":
                continue
            res.append({"name": n1, "time": t2 - t1})
        res.append({"name": "ALL", "time": last - first})
        return res

    def get_json_merged(self):
        timings = self.get_json()
        prev_name = None
        prev_time = 0
        merged_timings = []
        for timing in timings:
            if prev_name is None:
                prev_name = timing["name"]
                prev_time = timing["time"]
            elif prev_name == timing["name"]:
                prev_time += timing["time"]
            else:
                merged_timings.append({"name": prev_name, "time": prev_time})
                prev_name = timing["name"]
                prev_time = timing["time"]
        merged_timings.append({"name": prev_name, "time": prev_time})
        return merged_timings
