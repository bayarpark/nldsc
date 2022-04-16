#include <string>
#include "ldscalc.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

PYBIND11_MODULE(ldscore, m) {
    py::class_<LDScoreParams> ldp(m, "LDScoreParams");
    ldp.def(py::init());
    ldp.def_readwrite("bedfile", &LDScoreParams::bedfile)
        .def_readwrite("num_of_snp", &LDScoreParams::num_of_snp)
        .def_readwrite("num_of_org", &LDScoreParams::num_of_org)
        .def_readwrite("ld_wind", &LDScoreParams::ld_wind)
        .def_readwrite("positions", &LDScoreParams::positions)
        .def_readwrite("maf", &LDScoreParams::maf);

    m.def("calculate", &calculate);
}


