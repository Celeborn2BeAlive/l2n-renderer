#version 450

// Hybrid Generator described in  http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
// More details here: http://math.stackexchange.com/questions/337782/pseudo-random-number-generation-on-the-gpu
uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);    
}

uint LCGStep(uint z, uint A, uint C)
{
    return (A * z + C);    
}

float rand1_TausLCG(inout uvec4 state)
{
    state.x = TausStep(state.x, 13, 19, 12, 4294967294);
    state.y = TausStep(state.y, 2, 25, 4, 4294967288);
    state.z = TausStep(state.z, 3, 11, 17, 4294967280);
    state.w = LCGStep(state.w, 1664525, 1013904223);

    return 2.3283064365387e-10 * float(state.x ^ state.y ^ state.z ^ state.w);
}

vec2 rand2_TausLCG(inout uvec4 state)
{
    return vec2(rand1_TausLCG(state), rand1_TausLCG(state));
}