#ifndef TRITON_DEVICE_PRINT_H
#define TRITON_DEVICE_PRINT_H

#include "experiment/runtime/runtime/rt.h"
#include "stdio.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <type_traits>

#define LogBufferPaddingBytes 64
#define BlockMaxSize 16 * 1024
#define NumPhysicalPerLogical 3
#define VerifyBorder(nextField, maxBuf)                                                                                \
    if (nextField > maxBuf) {                                                                                          \
        std::cout << std::endl                                                                                         \
                  << "WARNING: out of bound! Print buffer size is " << BlockMaxSize << " - " << LogBufferPaddingBytes  \
                  << "Bytes which is exceeded!" << std::endl;                                                          \
        return;                                                                                                        \
    }
#define __gm__

namespace TTAscDebug
{

    enum NodeTy
    {
        END,
        NORMAL,
        FLOAT,
        INT,
        CHAR,
        STRING,
        POINTER
    };

    struct PrintPayloadData
    {
        __gm__ char* LogWholeRegion;
        unsigned     BlockNum;
        size_t       LogBufferSize;
        PrintPayloadData() : LogWholeRegion((__gm__ char*)nullptr), LogBufferSize(0), BlockNum(0) {}
    };

    struct DebugTunnelData
    {
        PrintPayloadData PrintData;
        DebugTunnelData() {}
    };

    void PrintFormatString(int8_t*& buf, int8_t* maxbuf)
    {
        // 读取长度
        short len = *reinterpret_cast<short*>(buf);
        buf += sizeof(len);

        // 检查缓冲区边界
        if (buf + len > maxbuf) {
            throw std::runtime_error("Buffer overflow");
        }

        // 获取格式字符串并输出
        const char* str = reinterpret_cast<const char*>(buf);
        std::cout << str; // 直接使用 std::cout 输出字符串

        // 移动缓冲区指针
        buf += len;
    }

    template <typename T>
    void PrintFormatString(int8_t*& buf, int8_t* maxbuf, T param)
    {
        // 读取长度
        short len = *reinterpret_cast<short*>(buf);
        buf += sizeof(len);

        // 确保不会越界
        if (buf + len > maxbuf) {
            throw std::runtime_error("Buffer overflow");
        }

        // 获取格式字符串
        const char* fmt = reinterpret_cast<const char*>(buf);
        buf += len;

        // 处理格式字符串并输出
        bool in_format = false;
        for (int i = 0; i < len; ++i) {
            if (fmt[i] == '%') {
                if (in_format) {
                    std::cout << '%'; // 遇到 %%
                    in_format = false;
                }
                else {
                    in_format = true;
                }
            }
            else {
                if (in_format) {
                    // 处理格式说明符
                    switch (fmt[i]) {
                    case 'd':
                    case 'i':
                        if constexpr (std::is_convertible_v<T, int>) {
                            std::cout << static_cast<int>(param);
                        }
                        else {
                            std::cerr << "Error: %d|i is invalid for typename" << std::endl;
                        }
                        break;
                    case 'u':
                        if constexpr (std::is_convertible_v<T, unsigned int>) {
                            std::cout << static_cast<unsigned int>(param);
                        }
                        else {
                            std::cerr << "Error: %u is invalid for typename" << std::endl;
                        }
                        break;
                    case 'f':
                    case 'F':
                        if constexpr (std::is_convertible_v<T, double>) {
                            std::cout << static_cast<double>(param);
                        }
                        else {
                            std::cerr << "Error: %f|F is invalid for typename" << std::endl;
                        }
                        break;
                    case 'e':
                    case 'E':
                        if constexpr (std::is_convertible_v<T, double>) {
                            std::cout << std::scientific << static_cast<double>(param);
                        }
                        else {
                            std::cerr << "Error: %e|E is invalid for typename" << std::endl;
                        }
                        break;
                    case 'g':
                    case 'G':
                        if constexpr (std::is_convertible_v<T, double>) {
                            std::cout << std::defaultfloat << static_cast<double>(param);
                        }
                        else {
                            std::cerr << "Error: %g|G is invalid for typename" << std::endl;
                        }
                        break;
                    case 'x':
                    case 'X':
                        if constexpr (std::is_convertible_v<T, int>) {
                            std::cout << std::hex << static_cast<int>(param);
                        }
                        else {
                            std::cerr << "Error: %x|X is invalid for typename" << std::endl;
                        }
                        break;
                    case 'o':
                        if constexpr (std::is_convertible_v<T, int>) {
                            std::cout << std::oct << static_cast<int>(param);
                        }
                        else {
                            std::cerr << "Error: %o is invalid for typename" << std::endl;
                        }
                        break;
                    case 'c':
                        if constexpr (std::is_convertible_v<T, char>) {
                            std::cout << static_cast<char>(param);
                        }
                        else {
                            std::cerr << "Error: %c is invalid for typename" << std::endl;
                        }
                        break;
                    case 's':
                        if constexpr (std::is_convertible_v<T, const char*>) {
                            std::cout << static_cast<const char*>(param);
                        }
                        else {
                            std::cerr << "Error: %s is invalid for typename" << std::endl;
                        }
                        break;
                    case 'p':
                        if constexpr (std::is_convertible_v<T, void*>) {
                            std::cout << reinterpret_cast<void*>(param);
                        }
                        else {
                            std::cerr << "Error: %p is invalid for typename" << std::endl;
                        }
                        break;
                    default:
                        std::cout << '%' << fmt[i]; // 无效格式说明符
                        break;
                    }
                    in_format = false;
                }
                else {
                    std::cout << fmt[i];
                }
            }
        }
    }

    void AnalyzeSerializedData(int8_t* buf, int logSize, int maxSize)
    {
        int8_t* bufEndAddr = buf + logSize;
        int8_t* maxbuf     = buf + maxSize;
        while (buf < bufEndAddr) {
            VerifyBorder((buf + sizeof(int8_t)), maxbuf);
            int8_t type = *(int8_t*)buf;
            while (type != NodeTy::END) {
                buf += sizeof(type);
                switch (type) {
                default:
                    break;
                case NodeTy::NORMAL: {
                    PrintFormatString(buf, maxbuf);
                    break;
                }
                case NodeTy::FLOAT: {
                    VerifyBorder((buf + sizeof(float)), maxbuf);
                    float param = *(float*)buf;
                    buf += sizeof(param);
                    PrintFormatString(buf, maxbuf, param);
                    break;
                }
                case NodeTy::INT: {
                    VerifyBorder((buf + sizeof(long long int)), maxbuf);
                    long long int param = *(long long int*)buf;
                    buf += sizeof(param);
                    PrintFormatString(buf, maxbuf, param);
                    break;
                }
                case NodeTy::STRING: {
                    VerifyBorder((buf + sizeof(short)), maxbuf);
                    short strlen = *(short*)buf;
                    buf += sizeof(strlen);
                    VerifyBorder((buf + strlen), maxbuf);
                    char* param = reinterpret_cast<char*>(buf);
                    buf += strlen;
                    PrintFormatString(buf, maxbuf, param);
                    break;
                }
                case NodeTy::CHAR: {
                    VerifyBorder((buf + sizeof(char)), maxbuf);
                    char param = *(char*)buf;
                    buf += sizeof(param);
                    PrintFormatString(buf, maxbuf, param);
                    break;
                }
                case NodeTy::POINTER: {
                    VerifyBorder((buf + 8), maxbuf);
                    void* param = *(void**)buf;
                    buf += sizeof(param);
                    PrintFormatString(buf, maxbuf, param);
                    break;
                }
                }
                VerifyBorder((buf + sizeof(int8_t)), maxbuf);
                type = *(int8_t*)buf;
            }
            buf += 1;
        }
    }

    void PrintBlock(char* Log, size_t LogBufferSize)
    {
        size_t LogSize = *reinterpret_cast<size_t*>(Log);
        if (LogSize > LogBufferSize || LogSize < 0) {
            std::cerr << " LOG SIZE ERROR !!!" << std::endl
                      << " log size needed = " << LogSize << ", buf size = " << LogBufferSize << std::endl;
            LogSize = LogBufferSize;
        }
        int8_t* Buf = reinterpret_cast<int8_t*>(Log + LogBufferPaddingBytes); // data addr
        AnalyzeSerializedData(Buf, LogSize, LogBufferSize);
        std::cout << std::endl;
    }

    void OnHostInitialize(PrintPayloadData* PrintData, unsigned BlockNum)
    {
        PrintData->LogBufferSize = BlockMaxSize;
        PrintData->BlockNum      = BlockNum;
        int WholeSize =
            (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum * NumPhysicalPerLogical;

        void*     Hbm_PrintPayloadData_start_addr = NULL;
        rtError_t error =
            rtMalloc(reinterpret_cast<void**>(&Hbm_PrintPayloadData_start_addr), WholeSize, RT_MEMORY_HBM, 0);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory for the printing function on the device side "
                         "fails to be allocated. As a result, the printing function fails!"
                      << std::endl;
            return;
        }
        PrintData->LogWholeRegion = (__gm__ char*)Hbm_PrintPayloadData_start_addr;

        char* hostMemIn;
        error = rtMallocHost(reinterpret_cast<void**>(&hostMemIn), WholeSize, 0);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory for the printing function on the host side "
                         "(before kernel) fails to be allocated. As a result, the "
                         "printing function fails!"
                      << std::endl;
            return;
        }
        for (int B = 0; B < (PrintData->BlockNum) * NumPhysicalPerLogical; B++) {
            char* blockStart     = hostMemIn + B * (PrintData->LogBufferSize + LogBufferPaddingBytes);
            *(size_t*)blockStart = 0; // initialize each counter to 0
        }
        error = rtMemcpy(PrintData->LogWholeRegion, WholeSize, hostMemIn, WholeSize, RT_MEMCPY_HOST_TO_DEVICE);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory copy from host to device fails, and the "
                         "printing function is invalid!"
                      << std::endl;
            return;
        }
        error = rtFreeHost(hostMemIn);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The host memory free (before kernel) of the device "
                         "print fails"
                      << std::endl;
            return;
        }
    }

    void OnHostFinish(PrintPayloadData* PrintData, rtStream_t Stream)
    {
        if (!PrintData->LogWholeRegion) {
            return;
        }
        std::size_t WholeSize =
            (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum * NumPhysicalPerLogical;
        char*     hostMemOut2;
        rtError_t error = rtMallocHost(reinterpret_cast<void**>(&hostMemOut2), WholeSize, 0);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory for the printing function on the device side "
                         "fails to be allocated. As a result, the printing function fails!"
                      << std::endl;
            return;
        }
        error = rtMemcpyAsync(hostMemOut2, WholeSize, PrintData->LogWholeRegion, WholeSize, RT_MEMCPY_DEVICE_TO_HOST,
                              Stream);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory copy of the device print on fails, and the "
                         "printing function is invalid!"
                      << std::endl;
            return;
        }
        error = rtStreamSynchronize(Stream);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: Synchronous waiting for the device print failed. The "
                         "printing function is invalid!"
                      << std::endl;
            return;
        }
        char*       outRaw2 = static_cast<char*>(hostMemOut2);
        const char* Line    = "-------------------------------------------------------";
        std::cout << Line << std::endl;
        std::cout << "---------------------------------HiIPU "
                     "Print---------------------------------"
                  << std::endl;
        std::cout << Line << std::endl;
        for (int B = 0; B < PrintData->BlockNum; B++) {
            std::cout << "==> Logical Block " << B << std::endl;
            char* cubeLog = outRaw2 + (PrintData->LogBufferSize + LogBufferPaddingBytes) * B;
            std::cout << "=> Physical Cube Block" << std::endl;
            PrintBlock(cubeLog, PrintData->LogBufferSize);

            int SubBlockDim = NumPhysicalPerLogical - 1;
            for (int V = 0; V < SubBlockDim; V++) {
                std::cout << "=> Physical Vector Block " << V << std::endl;
                char* VectorBase = outRaw2 + (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum;
                char* VectorLog  = VectorBase + (PrintData->LogBufferSize + LogBufferPaddingBytes) * SubBlockDim * B +
                                  (PrintData->LogBufferSize + LogBufferPaddingBytes) * V;
                PrintBlock(VectorLog, PrintData->LogBufferSize);
            }
        }
        error = rtFree(PrintData->LogWholeRegion);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The device memory free of the device print fails" << std::endl;
            return;
        }
        error = rtFreeHost(hostMemOut2);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The host memory free (after kernel) of the device "
                         "print fails"
                      << std::endl;
            return;
        }
    }

    DebugTunnelData* Open(unsigned BlockNum)
    {
        DebugTunnelData debugTunnelDataForHost;
        OnHostInitialize(&(debugTunnelDataForHost.PrintData), BlockNum);
        void*     Hbm_PrintPayloadData_start_addr = NULL;
        rtError_t error                           = rtMalloc(reinterpret_cast<void**>(&Hbm_PrintPayloadData_start_addr),
                                                             sizeof(debugTunnelDataForHost), RT_MEMORY_HBM, 0);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory for the printing function on the device side "
                         "fails to be allocated. As a result, the printing function fails!"
                      << std::endl;
            return nullptr;
        }
        if (Hbm_PrintPayloadData_start_addr == nullptr) {
            std::cout << "WARNING: failed to allocate DebugTunnelData memory" << std::endl;
            return nullptr;
        }
        error = rtMemcpy(Hbm_PrintPayloadData_start_addr, sizeof(debugTunnelDataForHost), &debugTunnelDataForHost,
                         sizeof(debugTunnelDataForHost), RT_MEMCPY_HOST_TO_DEVICE);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory copy of the device print on fails, and the "
                         "printing function is invalid!"
                      << std::endl;
            return nullptr;
        }
        return reinterpret_cast<DebugTunnelData*>(Hbm_PrintPayloadData_start_addr);
    }

    void Close(DebugTunnelData* DTData, rtStream_t Stream)
    {
        if (!DTData) {
            return;
        }
        DebugTunnelData debugTunnelDataForHost;
        rtError_t       error = rtStreamSynchronize(Stream);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: Synchronous waiting for the device print failed. The "
                         "printing function is invalid!"
                      << std::endl;
        }
        error = rtMemcpy(&debugTunnelDataForHost, sizeof(debugTunnelDataForHost), DTData,
                         sizeof(debugTunnelDataForHost), RT_MEMCPY_DEVICE_TO_HOST);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory copy of the device print on fails, and the "
                         "printing function is invalid!"
                      << std::endl;
            return;
        }
        OnHostFinish(&(debugTunnelDataForHost.PrintData), Stream);

        error = rtFree(DTData);
        if (error != RT_ERROR_NONE) {
            std::cerr << "ERROR: The memory free of the device print fails, and the "
                         "device print is invalid!"
                      << std::endl;
            return;
        }
        fflush(stdout);
    }

} // namespace TTAscDebug

#endif
