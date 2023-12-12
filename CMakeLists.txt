cmake_minimum_required(VERSION 3.10)
project(CompilerAutotuning)

set(LLVM_REPO_DIR "" CACHE PATH "Path to LLVM repository")
set(LLVM_VERSION "Your_LLVM_Version_Here" CACHE STRING "LLVM Version")

add_subdirectory(./Strategy/RL_strategy/obsUtility/)