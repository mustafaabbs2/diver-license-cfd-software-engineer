//
// helpers.h
//
// Copyright (c) 2024 dive solutions GmbH. All rights reserved.
//

#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using omp_iterator = long long;

namespace py = pybind11;

template <typename T>
class Span
{
public:
	using iterator = T*;
	using const_iterator = const T*;

	using reference = T&;
	using const_reference = const T&;

	using pointer = T*;
	using const_pointer = const T*;

	using value_type = T;

	Span() = default;
	Span(const Span<T>&) = default;
	Span(Span<T>&&) = default;
	~Span() = default;
	Span<T>& operator=(Span<T>&&) = default;
	Span<T>& operator=(const Span<T>&) = default;

	template <typename U = T, typename = typename std::enable_if<std::is_const_v<U>>>
	Span(const Span<std::remove_const_t<T>>& s) noexcept
		: sz_(s.size())
		, mem_(s.data())
	{}
	Span(const size_t n, T* data) noexcept
		: sz_(n)
		, mem_(data)
	{}

	Span(const std::vector<std::remove_const_t<T>>& vec) noexcept
		: sz_(vec.size())
		, mem_(vec.data())
	{}

	Span(std::vector<T>& vec) noexcept
		: sz_(vec.size())
		, mem_(vec.data())
	{}

	operator std::vector<std::remove_const_t<T>>() const
	{
		return std::vector<std::remove_const_t<T>>(begin(), end());
	}

	iterator begin() const
	{
		return mem_;
	}
	iterator end() const
	{
		return mem_ + sz_;
	}

	pointer data() const
	{
		return mem_;
	}

	bool empty() const
	{
		return sz_ == 0;
	}
	size_t size() const
	{
		return sz_;
	}

	reference operator[](const size_t idx) const
	{
		return mem_[idx];
	}

private:
	size_t sz_ = 0;
	T* mem_ = nullptr;
};

template <typename T>
pybind11::array CastVectorToNumpy(std::vector<T> cont)
{
	pybind11::buffer_info buf;
	buf.format = pybind11::format_descriptor<T>::format();
	buf.itemsize = sizeof(T);
	buf.ndim = 1;
	buf.shape = {(pybind11::ssize_t)cont.size()};
	buf.strides = {(pybind11::ssize_t)sizeof(T)};
	std::vector<T>* mptr = new std::vector<T>(std::move(cont));
	pybind11::capsule free_when_done(mptr, [](void* ptr) {
		std::vector<T>* pmem = reinterpret_cast<std::vector<T>*>(ptr);
		delete pmem;
	});
	buf.ptr = mptr->data();
	return pybind11::array(buf, free_when_done);
}

template <typename T>
Span<T> CastNumpyToSpan(const pybind11::array& obj)
{
	const pybind11::buffer_info buf = obj.request();
	if(buf.shape[0] > 0)
	{
		const size_t buf_ndim = static_cast<size_t>(buf.ndim);
		if(buf.itemsize != sizeof(T))
		{
			throw std::invalid_argument("Wrong data type! Array type expeceted to be " +
										std::to_string(sizeof(T)) + " bytes. Type found has " +
										std::to_string(buf.itemsize) + " bytes");
		}
		if(buf_ndim != 1)
		{
			throw std::invalid_argument("Number of dimensions is not 1 for a scalar type");
		}
		if(buf.strides[0] != sizeof(T))
		{
			throw std::invalid_argument("Wrong stride. Given scalar array is not C-contiguous");
		}
	}
	return Span<T>(buf.shape[0], reinterpret_cast<T*>(buf.ptr));
}