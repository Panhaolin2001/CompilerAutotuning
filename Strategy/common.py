from enum import Enum
import subprocess
import re
import os
import time

class Actions(Enum):
    AddDiscriminators = "--add-discriminators"
    Adce = "--adce"
    AlignmentFromAssumptions = "--alignment-from-assumptions"
    AlwaysInline = "--always-inline"
    Annotation2metadata = "--annotation2metadata"
    AssumeBuilder = "--assume-builder"
    AssumeSimplify = "--assume-simplify"
    AttributorCgscc = "--attributor-cgscc"
    Attributor = "--attributor"
    Barrier = "--barrier"
    Bdce = "--bdce"
    BreakCritEdges = "--break-crit-edges"
    Simplifycfg = "--simplifycfg"
    CallsiteSplitting = "--callsite-splitting"
    CalledValuePropagation = "--called-value-propagation"
    CanonFreeze = "--canon-freeze"
    Consthoist = "--consthoist"
    Constmerge = "--constmerge"
    CorrelatedPropagation = "--correlated-propagation"
    CrossDsoCfi = "--cross-dso-cfi"
    DfaJumpThreading = "--dfa-jump-threading"
    Deadargelim = "--deadargelim"
    Dce = "--dce"
    Dse = "--dse"
    DivRemPairs = "--div-rem-pairs"
    EarlyCseMemssa = "--early-cse-memssa"
    EarlyCse = "--early-cse"
    ElimAvailExtern = "--elim-avail-extern"
    FixIrreducible = "--fix-irreducible"
    Flattencfg = "--flattencfg"
    Float2int = "--float2int"
    Forceattrs = "--forceattrs"
    Inline = "--inline"
    GvnHoist = "--gvn-hoist"
    Gvn = "--gvn"
    Globaldce = "--globaldce"
    Globalopt = "--globalopt"
    Globalsplit = "--globalsplit"
    Hotcoldsplit = "--hotcoldsplit"
    Ipsccp = "--ipsccp"
    Iroutliner = "--iroutliner"
    Indvars = "--indvars"
    Irce = "--irce"
    InferAddressSpaces = "--infer-address-spaces"
    Inferattrs = "--inferattrs"
    InjectTliMappings = "--inject-tli-mappings"
    Instsimplify = "--instsimplify"
    Instcombine = "--instcombine"
    Instnamer = "--instnamer"
    JumpThreading = "--jump-threading"
    Lcssa = "--lcssa"
    Licm = "--licm"
    LibcallsShrinkwrap = "--libcalls-shrinkwrap"
    LoadStoreVectorizer = "--load-store-vectorizer"
    LoopDataPrefetch = "--loop-data-prefetch"
    LoopDeletion = "--loop-deletion"
    LoopDistribute = "--loop-distribute"
    LoopExtract = "--loop-extract"
    LoopFlatten = "--loop-flatten"
    LoopFusion = "--loop-fusion"
    LoopIdiom = "--loop-idiom"
    LoopInstsimplify = "--loop-instsimplify"
    LoopInterchange = "--loop-interchange"
    LoopLoadElim = "--loop-load-elim"
    LoopPredication = "--loop-predication"
    LoopReroll = "--loop-reroll"
    LoopRotate = "--loop-rotate"
    LoopSimplifycfg = "--loop-simplifycfg"
    LoopSimplify = "--loop-simplify"
    LoopSink = "--loop-sink"
    # LoopReduce = "--loop-reduce" # 两个叠加起来会报错
    LoopUnrollAndJam = "--loop-unroll-and-jam"
    LoopUnroll = "--loop-unroll"
    LoopVectorize = "--loop-vectorize"
    LoopVersioningLicm = "--loop-versioning-licm"
    LoopVersioning = "--loop-versioning"
    Loweratomic = "--loweratomic"
    LowerConstantIntrinsics = "--lower-constant-intrinsics"
    LowerExpect = "--lower-expect"
    LowerGlobalDtors = "--lower-global-dtors"
    LowerGuardIntrinsic = "--lower-guard-intrinsic"
    Lowerinvoke = "--lowerinvoke"
    LowerMatrixIntrinsicsMinimal = "--lower-matrix-intrinsics-minimal"
    LowerMatrixIntrinsics = "--lower-matrix-intrinsics"
    Lowerswitch = "--lowerswitch"
    LowerWidenableCondition = "--lower-widenable-condition"
    Memcpyopt = "--memcpyopt"
    Mergefunc = "--mergefunc"
    Mergeicmps = "--mergeicmps"
    MldstMotion = "--mldst-motion"
    NaryReassociate = "--nary-reassociate"
    Newgvn = "--newgvn"
    PartialInliner = "--partial-inliner"
    PartiallyInlineLibcalls = "--partially-inline-libcalls"
    FunctionAttrs = "--function-attrs"
    Mem2reg = "--mem2reg"
    Reassociate = "--reassociate"
    RedundantDbgInstElim = "--redundant-dbg-inst-elim"
    Reg2mem = "--reg2mem"
    RpoFunctionAttrs = "--rpo-function-attrs"
    RewriteStatepointsForGc = "--rewrite-statepoints-for-gc"
    Sccp = "--sccp"
    SlpVectorizer = "--slp-vectorizer"
    Sroa = "--sroa"
    ScalarizeMaskedMemIntrin = "--scalarize-masked-mem-intrin"
    Scalarizer = "--scalarizer"
    SeparateConstOffsetFromGep = "--separate-const-offset-from-gep"
    SimpleLoopUnswitch = "--simple-loop-unswitch"
    Sink = "--sink"
    SpeculativeExecution = "--speculative-execution"
    Slsr = "--slsr"
    StripDeadPrototypes = "--strip-dead-prototypes"
    StripDebugDeclare = "--strip-debug-declare"
    StripGcRelocates = "--strip-gc-relocates"
    StripNondebug = "--strip-nondebug"
    StripNonlinetableDebuginfo = "--strip-nonlinetable-debuginfo"
    Strip = "--strip"
    Structurizecfg = "--structurizecfg"
    Tlshoist = "--tlshoist"
    Tailcallelim = "--tailcallelim"
    Mergereturn = "--mergereturn"
    UnifyLoopExits = "--unify-loop-exits"
    VectorCombine = "--vector-combine"

# Set the global directory path for LLVM tools
llvm_tools_path = ""

def set_llvm_tools_path(bin_file):
    # Declare the global variable within the function
    global llvm_tools_path
    llvm_tools_path = bin_file

def compile_cpp_to_ll(input_cpp, log_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if log_dir is None:
        log_dir = os.path.join(script_dir, "log")

    os.makedirs(log_dir, exist_ok=True)
    output_ll = os.path.join(log_dir, "output.ll")

    cmd = [
        os.path.join(llvm_tools_path, "clang++"), "-S", "-emit-llvm",
        input_cpp, "-std=c++2a", "-o", output_ll
    ]
    subprocess.run(cmd, check=True)
    return output_ll

def GenerateBCFile(file_name, optimization_options):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "log")

    opt_path = os.path.join(llvm_tools_path, "opt")
    output_bc = os.path.join(log_dir, "output.bc")

    # 扁平化 optimization_options
    flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

    # 运行 opt 命令
    cmd_opt = [opt_path] + flat_opt_options + [file_name, "-o", output_bc]
    subprocess.run(cmd_opt, check=True)

    return output_bc

def GenerateObjFile(file_name, optimization_options):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "log")

    opt_path = os.path.join(llvm_tools_path, "opt")
    llc_path = os.path.join(llvm_tools_path, "llc")
    output_o = os.path.join(log_dir, "output.o")
    output_ll = os.path.join(log_dir, "output.bc")

    flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

    cmd_opt = [opt_path] + flat_opt_options + [file_name, "-o", output_ll]
    subprocess.run(cmd_opt, check=True)

    cmd_llc = [llc_path, "-filetype=obj", "-relocation-model=pic", "-o", output_o, output_ll]
    subprocess.run(cmd_llc, check=True)

    return output_o

def create_executable(object_files, output_file):
    clang_path = os.path.join(llvm_tools_path, "clang++")

    command = [clang_path, "-fPIE"] + object_files + ["-o", output_file]

    try:
        subprocess.run(command, check=True)

    except FileNotFoundError:
        print("Error: clang++ not found. Please check the clang_path variable.")
    except subprocess.CalledProcessError as e:
        print(f"Error: clang++ failed with exit code {e.returncode}")
        print(e.stderr)

def run_bitcode_and_get_time(executable_file):
    try:
        result = subprocess.run(
            [executable_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            error_message = result.stderr.strip()
            raise subprocess.CalledProcessError(result.returncode, executable_file, output=error_message)
        
        output = result.stdout

        match = re.search(r"Program average duration: (.+?)ns", output)
        if match:
            duration = float(match.group(1))
            return duration
        else:
            raise ValueError("Cannot find the program average duration in the output")
    except subprocess.TimeoutExpired:
        print(f"Execution of {executable_file} timed out after {60} seconds")
        return 999999999999
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running {executable_file}") from e

def get_codesize(ll_file, *opt_flags):
    # Convert the opt_flags into a list if it's a string
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Call the compile_with_optimizations function with the list of optimization flags
    object_file = GenerateObjFile(ll_file, opt_flags_list)
    executable_file = object_file.replace(".o", "")
    create_executable([object_file], executable_file)

    # Use objdump to get the size of the .text section
    cmd = ['objdump', '-h', executable_file]
    output = subprocess.check_output(cmd).decode('utf-8')
    
    # Extract the size of the .text section from the output
    for line in output.splitlines():
        if '.text' in line:
            size = line.split()[2]
            return int(size, 16)  # Convert hexadecimal to integer
    return 0

def get_pass_feature_internal(ll_file, *opt_flags, Arch="x86"):
    # Convert the opt_flags into a list if it's a string
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Assume GenerateObjFile and create_executable functions are already defined
    object_file = GenerateObjFile(ll_file, opt_flags_list)
    executable_file = object_file.replace(".o", "")
    create_executable([object_file], executable_file)

    # Use objdump to get the assembly code of the .text section
    cmd = ['objdump', '-d', '-j', '.text', executable_file]
    output = subprocess.check_output(cmd).decode('utf-8')

    # Extract assembly instructions using regex
    asm_lines = re.findall(r'\t(.+)$', output, re.MULTILINE)
    
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

def feature_change_due_to_pass(ll_file, *opt_flags):
    
    baseline_counts = get_pass_feature_internal(ll_file, "-O0")  # Get the counts for no optimizations
    pass_counts = get_pass_feature_internal(ll_file, *opt_flags)  # Get the counts for the given optimization flags
    
    # Compute and return the differences
    diffs = {}
    for key in baseline_counts:
        diffs[key] = pass_counts[key] - baseline_counts[key]
    
    return diffs

def get_instrcount(ll_file, *opt_flags):
    # Convert the opt_flags into a list if it's a string
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Call the compile_with_optimizations function with the list of optimization flags
    bc_file = GenerateBCFile(ll_file, opt_flags_list)

    # Use llvm-dis to convert .bc to .ll
    ll_file = bc_file.replace('.bc', '_isntr.ll')
    llvmdis_path = os.path.join(llvm_tools_path, "llvm-dis")
    subprocess.run([llvmdis_path, bc_file, '-o', ll_file], check=True)

    try:
        with open(ll_file, 'r') as f:
            content = f.read()
            # Count instructions: We'll consider lines with '=' as instructions for simplicity.
            # This is a basic heuristic and might not cover all cases.
            instr_count = content.count('=')
            return instr_count
    finally:
        # Clean up the intermediate .ll file
        if os.path.exists(ll_file):
            os.remove(ll_file)

def get_runtime_internal(ll_file, *opt_flags):
    # Convert the opt_flags into a list if it's a string
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Call the compile_with_optimizations function with the list of optimization flags
    object_file = GenerateObjFile(ll_file, opt_flags_list)
    executable_file = object_file.replace(".o", "")
    create_executable([object_file], executable_file)

    total_time = 0
    num_runs = 1000
    try:
        for _ in range(num_runs):
            # Measure start time
            start_time = time.time()
            # Run the executable
            subprocess.run([executable_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Measure end time and accumulate duration
            total_time += time.time() - start_time
        # Calculate average duration
        avg_duration = total_time / num_runs
        return avg_duration
    finally:
        # Clean up the intermediate files
        os.remove(object_file)
        os.remove(executable_file)

def get_runtime(ll_file, *opt_flags):
    # Convert the opt_flags into a list if it's a string
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Call the compile_with_optimizations function with the list of optimization flags
    object_file = GenerateObjFile(ll_file, opt_flags_list)
    executable_file = object_file.replace(".o", "")
    create_executable([object_file], executable_file)

    try:
        duration = run_bitcode_and_get_time(executable_file)
        return duration / 10e9
    finally:
        # Clean up the intermediate files
        os.remove(object_file)
        os.remove(executable_file)