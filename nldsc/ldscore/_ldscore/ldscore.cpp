#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#define ARMA_NO_DEBUG


#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ldscalc.h"
#include "data.h"

namespace py = pybind11;


PYBIND11_MODULE(_ldscore, m) {
    py::class_<LDScoreParams> LDScoreParams_(m, "LDScoreParams");
    LDScoreParams_.def(py::init());
    LDScoreParams_.def(
            py::init<const std::string&, int, int, double, double, double, double, const std::vector<double>&>(),
            py::arg("bfile"),
            py::kw_only(),
            py::arg("n_snp"),
            py::arg("n_org"),
            py::arg("ld_wind"),
            py::arg("maf"),
            py::arg("std_thr"),
            py::arg("rsq_thr"),
            py::arg("positions")
    );

    LDScoreParams_.def_readwrite("bedfile", &LDScoreParams::bedfile)
    .def_readwrite("n_snp", &LDScoreParams::n_snp)
    .def_readwrite("n_org", &LDScoreParams::n_org)
    .def_readwrite("ld_wind", &LDScoreParams::ld_wind)
    .def_readwrite("positions", &LDScoreParams::positions)
    .def_readwrite("maf", &LDScoreParams::maf)
    .def_readwrite("std_thr", &LDScoreParams::std_thr)
    .def_readwrite("rsq_thr", &LDScoreParams::rsq_thr);


    py::class_<LDScoreResult> LDScoreResult_(m, "LDScoreResult");
    LDScoreResult_.def(py::init());
    LDScoreResult_.def_readwrite("l2", &LDScoreResult::l2)
    .def_readwrite("l2d", &LDScoreResult::l2d)
    .def_readwrite("maf", &LDScoreResult::maf)
    .def_readwrite("residuals_std", &LDScoreResult::residuals_std)
    .def_readwrite("l2_ws", &LDScoreResult::l2_ws)
    .def_readwrite("l2d_ws", &LDScoreResult::l2d_ws)
    .def_readwrite("l2d_wse", &LDScoreResult::l2d_wse);

    m.def("calculate", &calculate);
}
