import re
import os
import io
import time
import subprocess
import tempfile

llvm_tools_path = ""
wafer_tools_path = ""

def set_llvm_tools_path(bin_file):
    global llvm_tools_path
    llvm_tools_path = bin_file
    print(f"-- Using LLVM Toolchain : {llvm_tools_path}")

def set_wafer_tools_path(bin_file):
    global wafer_tools_path
    wafer_tools_path = bin_file

def compile_cpp_to_ll(input_cpp, llvm_tools_path=None):
    if input_cpp.endswith(".ll") or input_cpp.endswith(".bc"):
        with open(input_cpp, 'r') as ll_file:
            return ll_file.read()

    clang_cmd = [
        os.path.join(llvm_tools_path, "clang++"), "-S", "-emit-llvm",
        input_cpp, "-std=c++2a", "-march=native", "-o", "-"
    ]

    clang_result = subprocess.run(clang_cmd, stdout=subprocess.PIPE, check=True)
    return clang_result.stdout.decode()

def GenerateOptimizedLLCode(input_code, optimization_options, llvm_tools_path=None):

    opt_path = os.path.join(llvm_tools_path, "opt")

    flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

    input_code_io = io.StringIO()
    input_code_io.write(input_code)
    input_code_io.seek(0)  # Reset the file position to the beginning

    cmd_opt = [opt_path] + flat_opt_options + ["-S", "-"]
    result = subprocess.run(cmd_opt, input=input_code_io.getvalue(), text=True, stdout=subprocess.PIPE, check=True)

    return result.stdout

def GenerateASMCode(input_code, optimization_options, llvm_tools_path=None):
    opt_path = os.path.join(llvm_tools_path, "opt")
    llc_path = os.path.join(llvm_tools_path, "llc")

    flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

    # Create a StringIO object and write the input code to it
    input_code_io = io.StringIO()
    input_code_io.write(input_code)
    input_code_io.seek(0)  # Reset the file position to the beginning

    # Use the StringIO object as input to opt tool
    cmd_opt = [opt_path] + flat_opt_options + ["-S", "-"]
    result_opt = subprocess.run(cmd_opt, input=input_code_io.getvalue(), text=True, stdout=subprocess.PIPE, check=True)

    # Use the optimized LLVM IR as input to llc tool
    cmd_llc = [llc_path, "-relocation-model=pic", "-mtriple=x86_64-unknown-linux-gnu", "-mattr=+sha", "-filetype=asm", "-o", "-"]
    result_llc = subprocess.run(cmd_llc, input=result_opt.stdout, text=True, stdout=subprocess.PIPE, check=True)

    return result_llc.stdout

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

def get_codesize(ll_code, *opt_flags, llvm_tools_path=None):
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Generate ASM code and create a temporary file to store it
    asm_code = GenerateASMCode(ll_code, opt_flags_list, llvm_tools_path)
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".s") as temp_asm_file:
        temp_asm_file.write(asm_code)
        temp_asm_file_path = temp_asm_file.name

    # Create an executable from the temporary ASM file
    executable_file = temp_asm_file_path.replace(".s", "")
    create_executable([temp_asm_file_path], executable_file, llvm_tools_path=llvm_tools_path)

    # Use objdump to get the size of the .text section
    cmd_objdump = ['objdump', '-h', executable_file]
    output = subprocess.check_output(cmd_objdump).decode('utf-8')

    for line in output.splitlines():
        if '.text' in line:
            size = line.split()[2]
            size_in_bytes = int(size, 16)

    # Clean up temporary files
    os.remove(temp_asm_file_path)
    os.remove(executable_file)

    return size_in_bytes
    
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

def get_runtime_internal(ll_code, *opt_flags, llvm_tools_path=None):
    opt_flags_list = opt_flags
    if len(opt_flags) == 1 and isinstance(opt_flags[0], str):
        opt_flags_list = opt_flags[0].split()

    # Create a temporary file to store ll code
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".ll") as temp_ll_file:
        temp_ll_file.write(ll_code)
        temp_ll_file_path = temp_ll_file.name

    # Create an executable from the temporary ASM file
    executable_file = temp_ll_file_path.replace(".s", "")
    create_executable([temp_ll_file_path], executable_file, llvm_tools_path=llvm_tools_path)

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
        os.remove(temp_ll_file)
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