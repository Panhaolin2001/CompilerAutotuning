import re
import os
import io
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

def compile_cpp_to_ll(input_cpp, is_wafer=False, wafer_lower_pass_options=None, llvm_tools_path=None, wafer_tools_path=None):
    if input_cpp.endswith(".ll") or input_cpp.endswith(".bc"):
        with open(input_cpp, 'r') as ll_file:
            return ll_file.read()

    clang_cmd = [
        os.path.join(llvm_tools_path, "clang++"), "-S", "-emit-llvm",
        input_cpp, "-std=c++2a", "-march=native", "-o", "-"
    ]

    if is_wafer:
        wafer_cmd = [
            os.path.join(wafer_tools_path, "wafer-frontend"), input_cpp, *wafer_lower_pass_options, "--wafer-to-llvmir", "-o", "-"
        ]

        clang_result = subprocess.run(clang_cmd, stdout=subprocess.PIPE, check=True)
        wafer_result = subprocess.run(wafer_cmd, input=clang_result.stdout, stdout=subprocess.PIPE, check=True)

        return wafer_result.stdout.decode()

    else:
        clang_result = subprocess.run(clang_cmd, stdout=subprocess.PIPE, check=True)
        return clang_result.stdout.decode()

def GenerateBCCode(input_code, optimization_options, llvm_tools_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "log")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    opt_path = os.path.join(llvm_tools_path, "opt")

    flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

    input_code_io = io.StringIO()
    input_code_io.write(input_code)
    input_code_io.seek(0)  # Reset the file position to the beginning

    cmd_opt = [opt_path] + flat_opt_options + ["-o", "-", "-"]
    result = subprocess.run(cmd_opt, input=input_code_io.getvalue().encode(), stdout=subprocess.PIPE, check=True)

    return result.stdout

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
    
    for line in output.splitlines():
        if '.text' in line:
            size = line.split()[2]
            return int(size, 16)

def get_instrcount(ll_code, *opt_flags, llvm_tools_path=None):
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Optimize LLVM IR code directly without generating intermediate files
    opt_path = os.path.join(llvm_tools_path, "opt")
    flat_opt_options = [str(item) for sublist in opt_flags_list for item in (sublist if isinstance(sublist, list) else [sublist])]

    cmd_opt = [opt_path] + flat_opt_options + ["-S", "-"]
    result = subprocess.run(cmd_opt, input=ll_code.encode(), stdout=subprocess.PIPE, check=True)

    # Count the occurrences of '=' in the optimized LLVM IR code
    instr_count = result.stdout.decode().count('=')
    
    return instr_count

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