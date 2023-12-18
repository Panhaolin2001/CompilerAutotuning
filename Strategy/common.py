import re
import os
import time
import subprocess
import threading

llvm_tools_path = ""
wafer_tools_path = ""

def set_llvm_tools_path(bin_file):
    global llvm_tools_path
    llvm_tools_path = bin_file
    print(f"-- Using LLVM Toolchain : {llvm_tools_path}")

def set_wafer_tools_path(bin_file):
    global wafer_tools_path
    wafer_tools_path = bin_file

def compile_cpp_to_ll(input_cpp, ll_file_dir=None, is_wafer=False,wafer_lower_pass_options=None, llvm_tools_path=None, wafer_tools_path=None):
    if input_cpp.endswith(".ll") or input_cpp.endswith(".bc"):
        return input_cpp
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if ll_file_dir is None:
        ll_file_dir = os.path.join(script_dir, "ll_file")

    os.makedirs(ll_file_dir, exist_ok=True)
    thread_id = threading.current_thread().ident
    output_ll = os.path.join(ll_file_dir, f"output_{thread_id}.ll")
    cpu_ll = os.path.join(ll_file_dir, f"cpu_{thread_id}.ll")
    out_ll = os.path.join(ll_file_dir, f"out_{thread_id}.ll")
    out1_ll = os.path.join(ll_file_dir, f"out1_{thread_id}.ll")

    if is_wafer:
        clang_cmd = [
            os.path.join(llvm_tools_path, "clang++"), "-S", "-emit-llvm",
            input_cpp, "-march=native", "-std=c++2a", "-o", output_ll
        ]
        subprocess.run(clang_cmd, check=True)

        wafer_cmd = [
            os.path.join(wafer_tools_path, "wafer-frontend"), input_cpp, *wafer_lower_pass_options, "--wafer-to-llvmir", "-o", cpu_ll
        ]
        subprocess.run(wafer_cmd, check=True)

        link_cmd = [
            os.path.join(llvm_tools_path, "llvm-link"), "-S", cpu_ll, output_ll, "-o", out_ll
        ]
        subprocess.run(link_cmd, check=True)

        return out_ll

    else:
        clang_cmd = [
            os.path.join(llvm_tools_path, "clang++"), "-S", "-emit-llvm",
            input_cpp, "-std=c++2a", "-march=native", "-o", output_ll
        ]
        subprocess.run(clang_cmd, check=True)

    return output_ll

def GenerateBCFile(file_name, optimization_options, llvm_tools_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "log")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    opt_path = os.path.join(llvm_tools_path, "opt")

    thread_id = threading.current_thread().ident
    output_bc = os.path.join(log_dir, f"output_{thread_id}.bc")

    flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

    cmd_opt = [opt_path] + flat_opt_options + [file_name, "-o", output_bc]
    subprocess.run(cmd_opt, check=True)

    return output_bc

def GenerateASMFile(file_name, optimization_options, llvm_tools_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "log")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    opt_path = os.path.join(llvm_tools_path, "opt")
    llc_path = os.path.join(llvm_tools_path, "llc")

    thread_id = threading.current_thread().ident
    output_s = os.path.join(log_dir, f"output_{thread_id}.s")
    output_ll = os.path.join(log_dir, f"output_{thread_id}.ll")

    flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

    cmd_opt = [opt_path, "-S"] + flat_opt_options + [file_name, "-o", output_ll]
    subprocess.run(cmd_opt, check=True)
    cmd_llc = [llc_path, "-relocation-model=pic", "-mtriple=x86_64-unknown-linux-gnu", "-mattr=+sha", "-filetype=asm", "-o", output_s, output_ll]
    subprocess.run(cmd_llc, check=True)

    return output_s

def create_executable(object_files, output_file, llvm_tools_path=None):
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

def get_codesize(ll_file, *opt_flags, llvm_tools_path=None):
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    asm_file = GenerateASMFile(ll_file, opt_flags_list, llvm_tools_path)
    executable_file = asm_file.replace(".s", "")
    create_executable([asm_file], executable_file, llvm_tools_path=llvm_tools_path)

    cmd = ['objdump', '-h', executable_file]
    output = subprocess.check_output(cmd).decode('utf-8')
    
    try:
        for line in output.splitlines():
            if '.text' in line:
                size = line.split()[2]
                return int(size, 16)
    finally:
        if os.path.exists(asm_file) and os.path.exists(executable_file) :
            os.remove(executable_file)
            os.remove(asm_file)
    return 0

def get_instrcount(ll_file, *opt_flags, llvm_tools_path=None):
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    bc_file = GenerateBCFile(ll_file, opt_flags_list, llvm_tools_path)
    ll_file = bc_file.replace('.bc', '_isntr.ll')
    llvmdis_path = os.path.join(llvm_tools_path, "llvm-dis")
    subprocess.run([llvmdis_path, bc_file, '-o', ll_file], check=True)

    try:
        with open(ll_file, 'r') as f:
            content = f.read()
            instr_count = content.count('=')
            return instr_count
    finally:
        if os.path.exists(ll_file) and os.path.exists(bc_file):
            os.remove(ll_file)
            os.remove(bc_file)

def get_runtime_internal(ll_file, *opt_flags, llvm_tools_path=None):
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    asm_file = GenerateASMFile(ll_file, opt_flags_list, llvm_tools_path)
    executable_file = asm_file.replace(".s", "")
    create_executable([asm_file], executable_file)

    total_time = 0
    num_runs = 1000
    try:
        for _ in range(num_runs):
            start_time = time.time()
            subprocess.run([executable_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            total_time += time.time() - start_time
        avg_duration = total_time / num_runs
        return avg_duration
    finally:
        os.remove(asm_file)
        os.remove(executable_file)

def get_runtime(ll_file, *opt_flags, llvm_tools_path=None):
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    asm_file = GenerateASMFile(ll_file, opt_flags_list, llvm_tools_path)
    executable_file = asm_file.replace(".o", "")
    create_executable([asm_file], executable_file)

    try:
        duration = run_bitcode_and_get_time(executable_file)
        return duration / 10e9
    finally:
        os.remove(asm_file)
        os.remove(executable_file)