#pragma once

#include <cstdint>

struct tinymt32_params
{
    uint32_t mat1;
    uint32_t mat2;
    uint32_t tmat;
};

extern tinymt32_params precomputed_tinymt_params[];

size_t precomputed_tinymt_params_count();