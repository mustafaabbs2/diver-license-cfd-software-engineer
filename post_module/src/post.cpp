//
// helpers.h
//
// Copyright (c) 2024 dive solutions GmbH. All rights reserved.
//

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <vector>

#include "helpers.h"

namespace py = pybind11;

void modify_array(pybind11::array& a_py) {
	Span<double> a = CastNumpyToSpan<double>(a_py);
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
	for (omp_iterator i = 0; i < a.size(); i++) {
		a[i] *= std::sin(a[i]);
		a[i] *= std::cos(a[i]);
	}
}

pybind11::array make_new_array(const pybind11::array& a_py, const double factor) {
	const Span<const double> a = CastNumpyToSpan<const double>(a_py);
	std::vector<double> b(a.size());
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
	for (omp_iterator i = 0; i < b.size(); i++) b[i] = a[i] * factor;
	return CastVectorToNumpy(b);
}

PYBIND11_MODULE(post_module, m) {
	m.def("add", [](int i, int j) { return i + j; });
	m.def("modify_array", &modify_array);
	m.def("make_new_array", &make_new_array);
}