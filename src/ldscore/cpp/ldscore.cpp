#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ldscalc.h"
#include "data.h"


namespace py = pybind11;

PYBIND11_MODULE(ldscore, m) {
    py::class_<LDScoreParams> LDScoreParams__(m, "LDScoreParams");
    LDScoreParams__.def(py::init());
    LDScoreParams__.def(py::init
            <const std::string&, int, int, double, double, double, const std::vector<double>&>
            ());
    LDScoreParams__.def_readwrite("bedfile", &LDScoreParams::bedfile)
        .def_readwrite("num_of_snp", &LDScoreParams::num_of_snp)
        .def_readwrite("num_of_org", &LDScoreParams::num_of_org)
        .def_readwrite("ld_wind", &LDScoreParams::ld_wind)
        .def_readwrite("positions", &LDScoreParams::positions)
        .def_readwrite("maf", &LDScoreParams::maf)
        .def_readwrite("std_threshold", &LDScoreParams::std_threshold);


    py::class_<LDScoreResult> LDScoreResult__(m, "LDScoreResult");
    LDScoreResult__.def(py::init());
    LDScoreResult__.def_readwrite("l2_add", &LDScoreResult::l2_add)
        .def_readwrite("l2_add", &LDScoreResult::l2_add)
        .def_readwrite("l2_nadd", &LDScoreResult::l2_nadd)
        .def_readwrite("mafs", &LDScoreResult::mafs)
        .def_readwrite("residuals_std", &LDScoreResult::residuals_std)
        .def_readwrite("additive_winsizes", &LDScoreResult::additive_winsizes)
        .def_readwrite("non_additive_winsizes", &LDScoreResult::non_additive_winsizes);

m.def("calculate", &calculate);

}
