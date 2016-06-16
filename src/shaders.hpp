#pragma once

#include <c2ba/glutils/GLProgram.hpp>

#include <unordered_map>
#include <filesystem>

namespace c2ba
{

namespace fs = std::experimental::filesystem;

// Use .at() to access elements since GLShader has no default constructor
using ShaderLibrary = std::unordered_map<fs::path, GLShader>;

GLShader loadShader(const fs::path& shaderPath);

void loadShaders(const fs::path& directoryPath, ShaderLibrary& shaderLibrary);

GLProgram compileProgram(const ShaderLibrary& shaderLibrary, std::vector<fs::path> shaderPaths);

template<typename... T>
GLProgram compileProgram(const ShaderLibrary& shaderLibrary, T&&... shaderPaths)
{
    return compileProgram(shaderLibrary, std::vector<fs::path>{shaderPaths...});
}

template<typename... T>
GLProgram compileProgram(const fs::path& rootDir, const ShaderLibrary& shaderLibrary, T&&... shaderPaths)
{
    return compileProgram(shaderLibrary, (rootDir / shaderPaths)...);
}

}

namespace std
{
template <> struct hash<c2ba::fs::path>
{
    size_t operator()(const c2ba::fs::path& x) const
    {
        using StringType = c2ba::fs::path::string_type;
        return hash<StringType>()(x.c_str());
    }
};
}