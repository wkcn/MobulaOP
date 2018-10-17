#ifndef MOBULA_INC_CTYPES_H_
#define MOBULA_INC_CTYPES_H_

namespace mobula {

typedef wchar_t wchar;
typedef char byte;
typedef unsigned char ubyte;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef long long longlong;
typedef unsigned long long ulonglong;
typedef long double longdouble;
typedef char* char_p;
typedef wchar_t* wchar_p;
typedef void* void_p;

#if defined(__x86_64__) || defined(_WIN64)
typedef uint64_t PointerValue;
#else
typedef uint32_t PointerValue;
#endif

}  // namespace mobula

#endif  // MOBULA_INC_CTYPES_H_
