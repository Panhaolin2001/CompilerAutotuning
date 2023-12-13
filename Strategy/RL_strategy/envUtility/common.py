from ...common import get_codesize, get_instrcount, GenerateASMFile, GenerateBCFile
from ..obsUtility.InstCount import get_inst_count_obs
from ..obsUtility.Autophase import get_autophase_obs
from ..obsUtility.IR2Vec import get_ir2vec_fa_obs, get_ir2vec_sym_obs
import re
import compiler_gym

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

def get_pass_feature_internal(ll_file, *opt_flags, obs_type="P2VInstCount", llvm_version="llvm-16.x",llvm_tools_path=None):
    
   if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
       opt_flags = opt_flags[0].split()
   bc_code = GenerateBCFile(ll_file, opt_flags,llvm_tools_path)

   if obs_type == "P2VInstCount":
       return get_inst_count_obs(bc_code, llvm_version)
   
   elif obs_type == "P2VAutoPhase":
       return get_autophase_obs(bc_code, llvm_version)
    
   elif obs_type == "P2VIR2VFa":
       if llvm_version == "llvm-10.0.0" or llvm_version == "llvm-14.x":
            return get_ir2vec_fa_obs(bc_code)
       else:
            raise ValueError(f"Unknown {llvm_version}, please choose 'llvm-14.x','llvm-10.x','llvm-10.0.0' on P2VIR2VFa ")
   
   elif obs_type == "P2VIR2VSym":
       if llvm_version == "llvm-10.0.0" or llvm_version == "llvm-14.x":
            return get_ir2vec_sym_obs(bc_code)
       else:
            raise ValueError(f"Unknown {llvm_version}, please choose 'llvm-14.x','llvm-10.x','llvm-10.0.0' on P2VIR2VSym ")
    
   elif obs_type == "P2VCustom":
       return pass2vec(ll_file, *opt_flags)
       
def feature_change_due_to_pass(ll_file, *opt_flags, baseline_counts, obs_type="P2VInstCount", llvm_version="llvm-16.x", llvm_tools_path=None):

    pass_counts = get_pass_feature_internal(ll_file,*opt_flags,obs_type=obs_type,llvm_version=llvm_version,llvm_tools_path=llvm_tools_path)  # Get the counts for the given optimization flags
    
    # Compute and return the differences
    diffs = {}
    for key in baseline_counts:
        diffs[key] = pass_counts[key] - baseline_counts[key]
    
    return diffs