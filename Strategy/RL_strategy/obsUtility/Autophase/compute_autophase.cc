#include <iostream>

#include "Autophase.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

struct AutophaseData {
    char name[64];
    int value;
};

const std::vector<std::string> AUTOPHASE_FEATURE_NAMES = {
    "BBNumArgsHi",
    "BBNumArgsLo",
    "onePred",
    "onePredOneSuc",
    "onePredTwoSuc",
    "oneSuccessor",
    "twoPred",
    "twoPredOneSuc",
    "twoEach",
    "twoSuccessor",
    "morePreds",
    "BB03Phi",
    "BBHiPhi",
    "BBNoPhi",
    "BeginPhi",
    "BranchCount",
    "returnInt",
    "CriticalCount",
    "NumEdges",
    "const32Bit",
    "const64Bit",
    "numConstZeroes",
    "numConstOnes",
    "UncondBranches",
    "binaryConstArg",
    "NumAShrInst",
    "NumAddInst",
    "NumAllocaInst",
    "NumAndInst",
    "BlockMid",
    "BlockLow",
    "NumBitCastInst",
    "NumBrInst",
    "NumCallInst",
    "NumGetElementPtrInst",
    "NumICmpInst",
    "NumLShrInst",
    "NumLoadInst",
    "NumMulInst",
    "NumOrInst",
    "NumPHIInst",
    "NumRetInst",
    "NumSExtInst",
    "NumSelectInst",
    "NumShlInst",
    "NumStoreInst",
    "NumSubInst",
    "NumTruncInst",
    "NumXorInst",
    "NumZExtInst",
    "TotalBlocks",
    "TotalInsts",
    "TotalMemInst",
    "TotalFuncs",
    "ArgsPhi",
    "testUnary"
};

extern "C" {
void GetAutophase(const char* irFilePath, struct AutophaseData* result) {
    llvm::SMDiagnostic error;
    llvm::LLVMContext ctx;

    auto module = llvm::parseIRFile(irFilePath, error, ctx);
    const auto features = autophase::InstCount::getFeatureVector(*module);

    if (features.size() > 0) {
      for(size_t i = 0; i < features.size(); ++i){
          strncpy(result[i].name, AUTOPHASE_FEATURE_NAMES[i].c_str(), sizeof(result[i].name));
          result[i].name[sizeof(result[i].name) - 1] = '\0'; // 确保字符串以 null 终止
          result[i].value = features[i];
      }
    }
}
}
