import itertools as it
from dataclasses import dataclass


def formatLines(elems, separator=" ", force_max=None):
    elems = [[str(x) for x in y] for y in elems]
    max_lens = [force_max or max(len(x) for x in y) for y in zip(*elems)]
    row_format = separator.join(f"{{: <{l}}}" for l in max_lens)
    return [row_format.format(*e) for e in elems]


@dataclass(frozen=True)
class Process:
    name: str
    is_signal: bool = False


@dataclass(frozen=True)
class Systematic:
    name: str
    dist: str


@dataclass(frozen=True)
class Channel:
    name: str


class DataCard:
    def __init__(self):
        self.processes = []
        self.systematics = []
        self.channels = []

        self.observations = {}
        self.process_systematics = {}
        self.process_rates = {}
        self.process_shapes = {}
        self.process_shape_systematics = {}

    def addProcess(self, process: Process):
        self.processes.append(process)

    def addSystematic(self, systematic: Systematic):
        self.systematics.append(systematic)

    def addChannel(self, channel: Channel):
        self.channels.append(channel)

    def setProcessSystematic(
        self, process: Process, systematic: Systematic, channel: Channel, value: float
    ):
        self.process_systematics[(channel, process, systematic)] = value

    def setProcessRate(self, process: Process, channel: Channel, value: float):
        self.process_rates[(channel, process)] = value

    def addShape(
        self,
        process: Process,
        channel: Channel,
        root_file,
        shape_name,
        shape_systematic_name,
    ):
        self.process_shapes[(channel, process)] = (
            root_file,
            shape_name,
            shape_systematic_name,
        )

    def addObservation(
        self,
        channel: Channel,
        root_file,
        hist_name,
        value,
    ):
        self.observations[channel] = (root_file, hist_name, value)

    def constructHeader(self):
        lines = []
        lines.append(f"# Autodatacard")
        lines.append(f"imax {len(self.channels)}")
        lines.append(f"jmax {len(self.processes) - 1}")
        lines.append(f"kmax {len(self.systematics)}")
        return lines

    def constructShapes(self):
        rows = []
        for (channel, process), (
            root_file,
            shape_name,
            shape_syst_name,
        ) in self.process_shapes.items():
            row = [
                "shapes",
                process.name,
                channel.name,
                root_file,
                shape_name,
                shape_syst_name,
            ]
            row = [x for x in row if x is not None]
            rows.append(row)

        for channel, (root_file, shape_name, value) in self.observations.items():
            row = ["shapes", "data_obs", channel.name, root_file, shape_name, ""]
            row = [x for x in row if x is not None]
            rows.append(row)

        return formatLines(rows, separator="  ")

    def constructObservations(self):
        cols = [["bin", "observation"]]
        for channel, (root_file, shape_name, value) in self.observations.items():
            cols.append([channel.name, "-1"])
        rows = list(zip(*cols))
        lines = formatLines(rows, separator="  ")
        return lines

    def constructSystematics(self):
        processes = enumerate(
            reversed(sorted(self.processes, key=lambda x: x.is_signal))
        )
        cols = []
        first_col = [
            "bin",
            "process",
            "process",
            "rate",
            *(x.name for x in self.systematics),
        ]
        second_col = ["", "", "", "", *(x.dist for x in self.systematics)]
        cols.append(first_col)
        cols.append(second_col)
        for channel, (i, process) in sorted(
            it.product(self.channels, processes), key=lambda x: x[0].name
        ):
            current_col = [
                channel.name,
                process.name,
                i,
                self.process_rates[(channel, process)],
            ]
            for systematic in self.systematics:
                s = self.process_systematics.get((channel, process, systematic), None)
                syst_string = str(s) if s is not None else "-"
                current_col.append(syst_string)
            cols.append(current_col)
        rows = list(zip(*cols))
        lines = formatLines(rows, separator="  ")
        lines.insert(4, "#" * len(lines[0]))
        return lines

    def dumps(self):
        lines = self.constructHeader()
        lines += [""] * 2
        lines += self.constructShapes()
        lines += [""] * 2
        lines += self.constructObservations()
        lines += [""] * 2
        lines += self.constructSystematics()
        output = "\n".join(lines) + "\n"
        return output


def main():
    bkg = Process("BackroundEstimate", False)
    sig = Process("Signal", True)
    b1 = Channel("bin1")
    b2 = Channel("bin2")
    card = DataCard()
    card.addChannel(b1)
    card.addChannel(b2)

    card.addProcess(sig)
    card.addProcess(bkg)

    s1 = Systematic("Lumi", "Normal")
    s2 = Systematic("ISR", "Normal")
    card.addSystematic(s1)
    card.addSystematic(s2)

    card.setProcessRate(sig, b1, 10)
    card.setProcessRate(sig, b2, 20)
    card.setProcessRate(bkg, b1, 100)
    card.setProcessRate(bkg, b2, 200)

    card.setProcessSystematic(sig, s1, b1, 1.1)
    card.setProcessSystematic(sig, s1, b2, 1.2)

    card.setProcessSystematic(sig, s2, b1, 1.1)
    card.setProcessSystematic(bkg, s2, b1, 2.1)
    card.setProcessSystematic(bkg, s2, b2, 0)

    card.addShape(sig, b1, "testfile.root", "sig", "sig_$SYSTEMATIC")
    card.addShape(bkg, b1, "testfile.root", "b_est", "b_est_$SYSTEMATIC")
    for i in range(0, 5):
        s = Systematic(f"b_est_EV_{i}_UP", "shape")
        card.addSystematic(s)
        card.setProcessSystematic(bkg, s, b1, 1)

        s = Systematic(f"b_est_EV_{i}_DOWN", "shape")
        card.addSystematic(s)
        card.setProcessSystematic(bkg, s, b1, 1)

    ret = card.dumps()
    print(ret)
    return


if __name__ == "__main__":
    main()
