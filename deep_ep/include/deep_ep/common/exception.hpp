// Copyright (c) 2026, Lu Lu
#pragma once

#include <exception>
#include <sstream>
#include <string>

// DeepEP-Ascend style assertions. Device builds trap, Host builds throw.
#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, ...) static_assert(cond, "" __VA_ARGS__)
#endif

class EPException : public std::exception {
    std::string message_;
public:
    EPException(const char* file, const int line, const std::string& error)
    {
        std::ostringstream stream;
        stream << "EP exception (" << file << ":" << line << "): " << error;
        message_ = stream.str();
    }
    const char* what() const noexcept override { return message_.c_str(); }
};

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond) \
    do { \
        if (!(cond)) throw EPException(__FILE__, __LINE__, #cond); \
    } while (0)
#endif

#ifndef EP_HOST_UNREACHABLE
#define EP_HOST_UNREACHABLE(reason) throw EPException(__FILE__, __LINE__, reason)
#endif

#ifndef EP_DEVICE_ASSERT
#if defined(__CCE__) || defined(__CCE_IS_AICORE__)
#define EP_DEVICE_ASSERT(cond) \
    do { if (!(cond)) { AscendC::Trap(); } } while (0)
#else
#define EP_DEVICE_ASSERT(cond) assert(cond)
#endif
#endif

