#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
        tailAddrToBlockSize.clear();
        headAddrToBlockSize.clear();
        freeBlocks.clear();
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);    // 如果size == 0，如何确认呢？
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        auto it = freeBlocks.lower_bound(freeBlockInfo{(size_t)0, size});
        if (it != freeBlocks.end()) {
            // 可以找到足够容纳size的块
            size_t addr = it->addr;
            size_t blockSize = it->size;
            freeBlocks.erase(it);
            headAddrToBlockSize.erase(addr);
            tailAddrToBlockSize.erase(addr + blockSize);
            if (blockSize > size) {
                freeBlocks.insert(freeBlockInfo{addr + size, blockSize - size});
                headAddrToBlockSize[addr + size] = blockSize - size;
                tailAddrToBlockSize[addr + blockSize] = blockSize - size;
            }
            this -> used += size;
            return addr;
        }
        else {
            // 找不到无法容纳size的块，扩展尾部长度
            if (tailAddrToBlockSize.find(this -> peak) != tailAddrToBlockSize.end()) {
                // 之前的尾部是空闲块，扩展这个块达到size大小
                size_t lastBlockSize = tailAddrToBlockSize[this -> peak];
                size_t lastBlockAddr = this -> peak - lastBlockSize;
                freeBlocks.erase(freeBlockInfo{lastBlockAddr, lastBlockSize});
                freeBlocks.insert(freeBlockInfo{lastBlockAddr, size});
                headAddrToBlockSize[lastBlockAddr] = size;
                tailAddrToBlockSize[lastBlockAddr + size] = size;
                this -> peak += size - lastBlockSize;
                this -> used += size;
                return lastBlockAddr;
            }
            else {
                // 之前的尾部是已被分配块，直接扩展size大小
                freeBlocks.insert(freeBlockInfo{this -> peak, size});
                headAddrToBlockSize[this -> peak] = size;
                tailAddrToBlockSize[this -> peak + size] = size;
                size_t retAddr = this -> peak;
                this -> peak += size;
                this -> used += size;
                return retAddr;
            }
        }
        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t tailAddr = addr + size;
        if (tailAddrToBlockSize.find(addr) != tailAddrToBlockSize.end()) {
            // 合并前面的块
            size_t blockSize = tailAddrToBlockSize[addr];
            size_t prevBlockAddr = addr - blockSize;
            freeBlocks.erase(freeBlockInfo{prevBlockAddr, blockSize});

            // 更新addr和size的信息，便于后面的合并
            addr = prevBlockAddr;
            size = blockSize + size;
            tailAddr = addr + size;
        }
        if (headAddrToBlockSize.find(tailAddr) != headAddrToBlockSize.end()) {
            // 合并后面的块
            size_t blockSize = headAddrToBlockSize[tailAddr];
            freeBlocks.erase(freeBlockInfo{tailAddr, blockSize});
        }
        //前后合并完成之后再插入当前的空余块
        freeBlocks.insert(freeBlockInfo{addr, size});
        headAddrToBlockSize[addr] = size;
        tailAddrToBlockSize[addr + size] = size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
