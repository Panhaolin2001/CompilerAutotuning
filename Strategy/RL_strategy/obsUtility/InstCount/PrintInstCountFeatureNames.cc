// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Print the name of each feature in the InstCount feature vector, in order.
#include <iostream>
#include <map>

#include "InstCount.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm_service;

struct Data {
    char name[64];
    int value;
};

extern "C" {
void GetInstCount(const char* irCode, struct Data* result) {
    llvm::SMDiagnostic error;
    llvm::LLVMContext ctx;

    // Create a memory buffer from the IR code string
    auto memBuffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(irCode), "", false);

    // Parse LLVM IR from the memory buffer
    auto module = llvm::parseIR(*memBuffer, error, ctx);

    if (!module) {
        // Handle parse error
        std::cerr << "Error parsing LLVM IR code: " << error.getMessage().str() << std::endl;
        return;
    }

    const auto features = InstCount::getFeatureVector(*module);
    const auto featureNames = InstCount::getFeatureNames();

    for (size_t i = 0; i < features.size(); ++i) {
        snprintf(result[i].name, sizeof(result[i].name), "%s", featureNames[i].c_str());
        result[i].value = features[i];
    }
}
}

