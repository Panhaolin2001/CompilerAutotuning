import subprocess
import re
import os
import time

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