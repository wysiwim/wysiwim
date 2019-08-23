import os

class ExperimentReport:
    def __init__(self):
        self.run = -1
        self.lines = []
        self.dataset = ""
        self.vis = ""
        self.algo = ""
        self.output_file = None

    def add_result(self, a, p, r, f1, alt_alg = None):
        line = "%s,%s,%s,%s,%s,%s,%s,%s\n" % (
            self.run, alt_alg if alt_alg else self.alg, self.dataset, self.vis, a, p, r, f1
        )
        self.lines.append(line)

        if self.output_file is not None:
            with open(self.output_file, "a+") as outf:
                outf.write(line)

    def save_to_file(self, filename):
        with open(filename, "w+") as outf:
            outf.write("run,algorithm,dataset,visualization,accuracy,precision,recall,f1\n")
            for l in self.lines:
                outf.write(l)

    def set_save_file(self, filename):
        self.output_file = filename
        if self.output_file is not None:
            with open(self.output_file, "w+") as outf:
                outf.write("run,algorithm,dataset,visualization,accuracy,precision,recall,f1\n")


    def reset(self):
        self.lines = []
        self.output_file = None


    def get_full_outpath(self):
        return os.getcwd()+"/"+self.output_file


report = ExperimentReport()

