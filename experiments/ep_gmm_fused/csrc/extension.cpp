// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Modified by zhu-mingzhe71 2026

#define Py_LIMITED_API_VERSION 0x03080000
#include <Python.h>

extern "C" PyObject* PyInit__C(void)
{
    static struct PyModuleDef moduleDef = {
        PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr,
    };
    return PyModule_Create(&moduleDef);
}
