
from nonlinear_benchmarks.benchmarks import CED, Cascaded_Tanks, EMPS, Silverbox, WienerHammerBenchMark
from nonlinear_benchmarks.utilities import Input_output_data
import nonlinear_benchmarks.error_metrics
import nonlinear_benchmarks.not_splitted_benchmarks

all_splitted_benchmarks = [CED, EMPS, Cascaded_Tanks, Silverbox, WienerHammerBenchMark]
all_not_splitted_benchmarks = [nonlinear_benchmarks.not_splitted_benchmarks.BoucWen, \
                               nonlinear_benchmarks.not_splitted_benchmarks.WienerHammerstein_Process_Noise, \
                               nonlinear_benchmarks.not_splitted_benchmarks.ParWHF, \
                               nonlinear_benchmarks.not_splitted_benchmarks.F16,\
                               nonlinear_benchmarks.not_splitted_benchmarks.Industrial_robot]
all_benchmarks = all_splitted_benchmarks + all_not_splitted_benchmarks