import ctypes
import os

class AutophaseDataStruct(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 64), ("value", ctypes.c_int)]

def get_autophase_obs(ir_file_path, llvm_version="llvm-16.x"):
    project_directory = os.path.dirname(os.path.abspath(__file__))
    library_path = None
    if llvm_version == "llvm-16.x":
        library_path = os.path.join(project_directory, '../../../build/Strategy/RL_strategy/obsUtility/Autophase', 'libAutophase_16_x.so')
    elif llvm_version == "llvm-14.x":
        library_path = os.path.join(project_directory, '../../../build/Strategy/RL_strategy/obsUtility/Autophase', 'libAutophase_14_x.so')
    elif llvm_version == "llvm-10.0.0":
        library_path = os.path.join(project_directory, '../../../build/Strategy/RL_strategy/obsUtility/Autophase', 'libAutophase_10_0_0.so')
    
    result_array = (AutophaseDataStruct * 56)()
    autophase_lib = ctypes.CDLL(library_path)

    autophase_lib.GetAutophase(ir_file_path.encode(), result_array)
    result_dict = {item.name.decode(): item.value for item in result_array}
    
    return result_dict
