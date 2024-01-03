from ..obsUtility.InstCount import get_inst_count_obs
from ..obsUtility.Autophase import get_autophase_obs
from ..obsUtility.IR2Vec import get_ir2vec_fa_obs, get_ir2vec_sym_obs
import re
import tempfile
import os

def get_pass_feature_internal(ll_code, *opt_flags, obs_type="P2VInstCount", llvm_version="llvm-16.x",llvm_tools_path=None):
    
   if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
       opt_flags = opt_flags[0].split()

   if "InstCount" in obs_type :
       return get_inst_count_obs(ll_code, llvm_version)
   
   elif "AutoPhase" in obs_type :
       return get_autophase_obs(ll_code, llvm_version)
    
   elif "IR2VecFa" in obs_type :
       if llvm_version == "llvm-10.0.0" or llvm_version == "llvm-14.x":
            return get_ir2vec_fa_obs(ll_code)
       else:
            raise ValueError(f"Unknown {llvm_version}, please choose 'llvm-14.x','llvm-10.x','llvm-10.0.0' on P2VIR2VFa ")
   
   elif "IR2VecSym" in obs_type:
       if llvm_version == "llvm-10.0.0" or llvm_version == "llvm-14.x":
            return get_ir2vec_sym_obs(ll_code)
       else:
            raise ValueError(f"Unknown {llvm_version}, please choose 'llvm-14.x','llvm-10.x','llvm-10.0.0' on P2VIR2VSym ")
    
def feature_change_due_to_pass(ll_code, *opt_flags, baseline_counts, obs_type="P2VInstCount", llvm_version="llvm-16.x", llvm_tools_path=None):

    pass_counts = get_pass_feature_internal(ll_code,*opt_flags,obs_type=obs_type,llvm_version=llvm_version,llvm_tools_path=llvm_tools_path)  # Get the counts for the given optimization flags
    
    # Compute and return the differences
    diffs = {}
    for key in baseline_counts:
        diffs[key] = pass_counts[key] - baseline_counts[key]
    
    return diffs