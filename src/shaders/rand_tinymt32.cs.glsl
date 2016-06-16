#version 450

// Source for tinyMT: https://github.com/MersenneTwister-Lab/TinyMT

#define TINYMT32_MEXP 127
#define TINYMT32_SH0 1
#define TINYMT32_SH1 10
#define TINYMT32_SH8 8
#define TINYMT32_MASK uint(0x7fffffff)
#define TINYMT32_MUL (1.0f / 16777216.0f)

/**
* tinymt32 internal state vector and parameters
*/
struct tinymt32_t
{
    uvec4 status;
    uint mat_1;
    uint mat_2;
    uint tmat;
	uint _align0;
};

/**
* This function changes internal state of tinymt32.
* Users should not call this function directly.
* @param random tinymt internal status
*/
void tinymt32_next_state(inout tinymt32_t random) {
    uint y = random.status[3];
    uint x = (random.status[0] & TINYMT32_MASK)
        ^ random.status[1]
        ^ random.status[2];
    x ^= (x << TINYMT32_SH0);
    y ^= (y >> TINYMT32_SH0) ^ x;
    random.status[0] = random.status[1];
    random.status[1] = random.status[2];
    random.status[2] = x ^ (y << TINYMT32_SH1);
    random.status[3] = y;
    random.status[1] ^= -(int(y & 1)) & random.mat_1;
    random.status[2] ^= -(int(y & 1)) & random.mat_2;
}

/**
* This function outputs 32-bit unsigned integer from internal state.
* Users should not call this function directly.
* @param random tinymt internal status
* @return 32-bit unsigned pseudorandom number
*/
uint tinymt32_temper(tinymt32_t random) 
{
    uint t0, t1;
    t0 = random.status[3];
#if defined(LINEARITY_CHECK)
    t1 = random->status[0]
        ^ (random->status[2] >> TINYMT32_SH8);
#else
    t1 = random.status[0]
        + (random.status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    t0 ^= -(int(t1 & 1)) & random.tmat;
    return t0;
}

/**
* This function outputs floating point number from internal state.
* Users should not call this function directly.
* @param random tinymt internal status
* @return floating point number r (1.0 <= r < 2.0)
*/
float tinymt32_temper_conv(tinymt32_t random) {
    uint t0, t1;
    uint u;

    t0 = random.status[3];
#if defined(LINEARITY_CHECK)
    t1 = random.status[0]
        ^ (random.status[2] >> TINYMT32_SH8);
#else
    t1 = random.status[0]
        + (random.status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    u = ((t0 ^ (-(int(t1 & 1)) & random.tmat)) >> 9)
        | uint(0x3f800000);
    return uintBitsToFloat(u);
}

/**
* This function outputs floating point number from internal state.
* Users should not call this function directly.
* @param random tinymt internal status
* @return floating point number r (1.0 < r < 2.0)
*/
float tinymt32_temper_conv_open(tinymt32_t random) {
    uint t0, t1;
    uint u;

    t0 = random.status[3];
#if defined(LINEARITY_CHECK)
    t1 = random.status[0]
        ^ (random.status[2] >> TINYMT32_SH8);
#else
    t1 = random.status[0]
        + (random.status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    u = ((t0 ^ (-(int(t1 & 1)) & random.tmat)) >> 9)
        | uint(0x3f800001);
    return uintBitsToFloat(u);
}

/**
* This function outputs 32-bit unsigned integer from internal state.
* @param random tinymt internal status
* @return 32-bit unsigned integer r (0 <= r < 2^32)
*/
uint tinymt32_generate_uint32(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper(random);
}

/**
* This function outputs floating point number from internal state.
* This function is implemented using multiplying by (1 / 2^24).
* floating point multiplication is faster than using union trick in
* my Intel CPU.
* @param random tinymt internal status
* @return floating point number r (0.0 <= r < 1.0)
*/
float tinymt32_generate_float(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return (tinymt32_temper(random) >> 8) * TINYMT32_MUL;
}

/**
* This function outputs floating point number from internal state.
* This function is implemented using union trick.
* @param random tinymt internal status
* @return floating point number r (1.0 <= r < 2.0)
*/
float tinymt32_generate_float12(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv(random);
}

/**
* This function outputs floating point number from internal state.
* This function is implemented using union trick.
* @param random tinymt internal status
* @return floating point number r (0.0 <= r < 1.0)
*/
float tinymt32_generate_float01(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv(random) - 1.0f;
}

/**
* This function outputs floating point number from internal state.
* This function may return 1.0 and never returns 0.0.
* @param random tinymt internal status
* @return floating point number r (0.0 < r <= 1.0)
*/
float tinymt32_generate_floatOC(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return 1.0f - tinymt32_generate_float(random);
}

/**
* This function outputs floating point number from internal state.
* This function returns neither 0.0 nor 1.0.
* @param random tinymt internal status
* @return floating point number r (0.0 < r < 1.0)
*/
float tinymt32_generate_floatOO(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv_open(random) - 1.0f;
}

/**
* This function outputs double precision floating point number from
* internal state. The returned value has 32-bit precision.
* In other words, this function makes one double precision floating point
* number from one 32-bit unsigned integer.
* @param random tinymt internal status
* @return floating point number r (0.0 < r <= 1.0)
*/
double tinymt32_generate_32double(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return double(tinymt32_temper(random)) * (1.0 / 4294967296.0);
}