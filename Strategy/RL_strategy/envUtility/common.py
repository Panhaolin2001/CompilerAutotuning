# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import NamedTuple, Optional
from ...common import get_codesize, get_instrcount, GenerateASMFile, create_executable
import subprocess
import re

class Pass(NamedTuple):
    """The declaration of an LLVM pass."""

    # The name of the pass, e.g. "AddDiscriminatorsPass".
    name: str
    # The opt commandline flag which turns this pass on, e.g. "-add-discriminators".
    flag: str
    # The docstring for this pass, as reported by `opt -help`. E.g. "Add DWARF path discriminators".
    description: str
    # The path of the C++ file which defines this pass, relative to the LLVM source tree root.
    source: str
    # The path of the C++ header which declares this pass, relative to the LLVM source tree root.
    # If the header path could not be inferred, this is None.
    header: Optional[str]
    # Boolean flags set in INITIALIZE_PASS().
    cfg: bool
    is_analysis: bool


def pass2vec(ll_file, *opt_flags, Arch="x86"):
     # Convert the opt_flags into a list if it's a string
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Assume GenerateObjFile and create_executable functions are already defined
    asm_file = GenerateASMFile(ll_file, opt_flags_list)

    # Extract assembly instructions using regex
    asm_lines = re.findall(r'\t(.+)$', asm_file, re.MULTILINE)
    
    if(Arch == "x86"):
        # Initialize instruction counters
        instruction_counters = {
            'total': 0,
            'vector': 0,
            'jump': 0,
            'add': 0,
            'sub': 0,
            'mul': 0,
            'div': 0,
            'inc': 0,
            'dec': 0,
            'neg': 0,
            'lea': 0,
            'imul': 0,
            'call': 0,
            'ret': 0,
            'and': 0,
            'or': 0,
            'xor': 0,
            'not': 0,
            'shl': 0,
            'shr': 0,
            'mov': 0,
            'pop': 0,
            'push': 0,
            'vector_load_store': 0,
            'Codesize' : 0,
            # 'Performance' : 0,
            'IrCount' : 0
        }

        # Compute instruction counts
        for line in asm_lines:
            instruction_counters['total'] += 1
            for keyword, count in instruction_counters.items():
                if re.search(fr'\b{keyword}\b', line):
                    instruction_counters[keyword] += 1

            # Special regex checks
            if re.search(r'\bv\w*', line):
                instruction_counters['vector'] += 1
            if re.search(r'(vmov|vpbroadcast|vgather|vscatter)', line):
                instruction_counters['vector_load_store'] += 1
            if re.search(r'\b(jmp|je|jne|jg|jge|jl|jle|ja|jae|jb|jbe|jz|jnz)\b', line):
                instruction_counters['jump'] += 1
            if re.search(r'\b(call|callq)\b', line):
                instruction_counters['call'] += 1
            if re.search(r'\b(ret|retq)\b', line):
                instruction_counters['ret'] += 1
        
        instruction_counters['Codesize'] = get_codesize(ll_file, *opt_flags)
        # instruction_counters['Performance'] = get_runtime(ll_file, *opt_flags)
        instruction_counters['IrCount'] = get_instrcount(ll_file, *opt_flags)

    # Return results
    return instruction_counters

def ir2vec(ll_file):
    pass

def get_pass_feature_internal(ll_file, *opt_flags, obs_type="pass2vec"):
   if obs_type == "pass2vec":
       return pass2vec(ll_file, *opt_flags, Arch="x86")
   elif obs_type == "ir2vec":
       pass
   elif obs_type == "pass2vec":
       pass

def feature_change_due_to_pass(ll_file, *opt_flags, obs_type="pass2vec"):
    
    baseline_counts = get_pass_feature_internal(ll_file, "-O0", obs_type="pass2vec")  # Get the counts for no optimizations
    pass_counts = get_pass_feature_internal(ll_file, *opt_flags, obs_type="pass2vec")  # Get the counts for the given optimization flags
    
    # Compute and return the differences
    diffs = {}
    for key in baseline_counts:
        diffs[key] = pass_counts[key] - baseline_counts[key]
    
    return diffs