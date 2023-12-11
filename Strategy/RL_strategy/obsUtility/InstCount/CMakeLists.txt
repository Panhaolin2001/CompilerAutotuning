include(ExternalProject)

# Add your source files
set(SOURCE_FILES
    InstCount.cc
    PrintInstCountFeatureNames.cc
)

set(LLVM_REPO_DIR "" CACHE PATH "Path to LLVM repository")
set(LLVM_VERSION "Your_LLVM_Version_Here" CACHE STRING "LLVM Version")


if(LLVM_VERSION STREQUAL "14")
    # Check for LLVM 14
    find_package(LLVM 14 REQUIRED CONFIG)
    set(LLVM_BUILD_INCLUDE_DIRS "")
    foreach(dir ${LLVM_INCLUDE_DIRS})
        if(dir MATCHES "/build*")
            list(APPEND LLVM_BUILD_INCLUDE_DIRS ${dir})
        endif()
    endforeach()

    if(LLVM_FOUND)

        message(STATUS "Found LLVM 14: ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

        # 存储 LLVM 14 的信息
        set(LLVM_14_INCLUDE_DIRS ${LLVM_BUILD_INCLUDE_DIRS})
        set(LLVM_14_DEFINITIONS ${LLVM_DEFINITIONS})
        set(LLVM_COMMIT_HASH "f28c006a5895")

        ExternalProject_Add(
            llvm-14.x
            PREFIX ${CMAKE_BINARY_DIR}/external
            GIT_REPOSITORY ${LLVM_REPO_DIR}
            GIT_TAG ${LLVM_COMMIT_HASH}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )

        # Map LLVM components to library names
        llvm_map_components_to_libnames(LLVM_LIBS core support bitwriter passes)
        message(STATUS "LLVM Component Libraries: ${LLVM_LIBS}")
        set(LLVM_14_LIBS ${LLVM_LIBS})

        # Add your source files and create a library
        add_library(InstCount_14_x SHARED ${SOURCE_FILES})

        target_include_directories(InstCount_14_x PRIVATE ${LLVM_14_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/external/src/llvm-14.x/llvm/include)
        target_compile_definitions(InstCount_14_x PRIVATE ${LLVM_14_DEFINITIONS})

        # Link against LLVM libraries
        target_link_libraries(InstCount_14_x ${LLVM_LIBS})

        # Specify additional libraries if needed (e.g., pthread, ncurses, etc.)
        target_link_libraries(InstCount_14_x pthread ncurses)

        # Set compiler flags as needed (e.g., -fno-rtti)
        target_compile_options(InstCount_14_x PRIVATE -fno-rtti)

    endif()

elseif(LLVM_VERSION STREQUAL "16")

    # Check for LLVM 16
    find_package(LLVM 16 REQUIRED CONFIG)
    set(LLVM_BUILD_INCLUDE_DIRS "")
    foreach(dir ${LLVM_INCLUDE_DIRS})
        if(dir MATCHES "/build*")
            list(APPEND LLVM_BUILD_INCLUDE_DIRS ${dir})
        endif()
    endforeach()

    if(LLVM_FOUND)
        message(STATUS "Found LLVM 16: ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

        set(LLVM_16_INCLUDE_DIRS ${LLVM_BUILD_INCLUDE_DIRS})
        set(LLVM_16_DEFINITIONS ${LLVM_DEFINITIONS})
        set(LLVM_COMMIT_HASH "7cbf1a2591520c2491aa35339f227775f4d3adf6")  # 替换为你要检出的 commit 的哈希值

        ExternalProject_Add(
            llvm-16.x
            PREFIX ${CMAKE_BINARY_DIR}/external
            GIT_REPOSITORY ${LLVM_REPO_DIR}
            GIT_TAG ${LLVM_COMMIT_HASH}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )

        # Map LLVM components to library names
        llvm_map_components_to_libnames(LLVM_LIBS core support bitwriter)
        set(LLVM_16_LIBS ${LLVM_LIBS})
        message(STATUS "LLVM Component Libraries: ${LLVM_16_LIBS}")

        # Add your source files and create a library
        add_library(InstCount_16_x SHARED ${SOURCE_FILES})
        
        target_include_directories(InstCount_16_x PRIVATE ${LLVM_16_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/external/src/llvm-16.x/llvm/include)
        target_compile_definitions(InstCount_16_x PRIVATE ${LLVM_16_DEFINITIONS})

        # Link against LLVM libraries
        target_link_libraries(InstCount_16_x ${LLVM_16_LIBS})

        # Specify additional libraries if needed (e.g., pthread, ncurses, etc.)
        target_link_libraries(InstCount_16_x pthread ncurses)

        # Set compiler flags as needed (e.g., -fno-rtti)
        target_compile_options(InstCount_16_x PRIVATE -fno-rtti)

    endif()

elseif(LLVM_VERSION STREQUAL "10.0.0")

    # Check for LLVM 10.0.0
    find_package(LLVM 10.0.0 REQUIRED CONFIG)
    set(LLVM_BUILD_INCLUDE_DIRS "")
    foreach(dir ${LLVM_INCLUDE_DIRS})
        if(dir MATCHES "/build*")
            list(APPEND LLVM_BUILD_INCLUDE_DIRS ${dir})
        endif()
    endforeach()

    if(LLVM_FOUND)
        message(STATUS "Found LLVM 10.0.0: ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

        set(LLVM_10_0_0_INCLUDE_DIRS ${LLVM_BUILD_INCLUDE_DIRS})
        set(LLVM_10_0_0_DEFINITIONS ${LLVM_DEFINITIONS})
        set(LLVM_COMMIT_HASH "d32170dbd5b0d54436537b6b75beaf44324e0c28")  # 替换为你要检出的 commit 的哈希值

        ExternalProject_Add(
            llvm-10.0.0
            PREFIX ${CMAKE_BINARY_DIR}/external
            GIT_REPOSITORY ${LLVM_REPO_DIR}
            GIT_TAG ${LLVM_COMMIT_HASH}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )

        # Map LLVM components to library names
        llvm_map_components_to_libnames(LLVM_LIBS core support bitwriter passes)
        set(LLVM_10_0_0_LIBS ${LLVM_LIBS})
        message(STATUS "LLVM Component Libraries: ${LLVM_10_0_0_LIBS}")

        # Add your source files and create a library
        add_library(InstCount_10_0_0 SHARED ${SOURCE_FILES})
        
        target_include_directories(InstCount_10_0_0 PRIVATE ${LLVM_10_0_0_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/external/src/llvm-10.0.0/llvm/include)
        target_compile_definitions(InstCount_10_0_0 PRIVATE ${LLVM_10_0_0_DEFINITIONS})

        # Link against LLVM libraries
        target_link_libraries(InstCount_10_0_0 ${LLVM_10_0_0_LIBS})

        # Specify additional libraries if needed (e.g., pthread, ncurses, etc.)
        target_link_libraries(InstCount_10_0_0 pthread ncurses)

        # Set compiler flags as needed (e.g., -fno-rtti)
        target_compile_options(InstCount_10_0_0 PRIVATE -fno-rtti)

    endif()
else()
    message(FATAL_ERROR "Unsupported LLVM version: ${LLVM_VERSION}")
endif()