import IR2Vec as i2v
import tempfile
import os

def save_ir_code_to_temp_file(ir_code):
    # Create a temporary file to store the LLVM IR code
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.ll', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(ir_code)

    return temp_file_path

def get_ir2vec_fa_obs(ir_code):
    # Save the LLVM IR code to a temporary file
    temp_file_path = save_ir_code_to_temp_file(ir_code)

    try:
        # Call i2v.generateEmbeddings with the temporary file path
        emb = i2v.generateEmbeddings(temp_file_path, "fa", "p")

        # Process the embeddings as needed
        program_dict = {index: value for index, value in enumerate(emb["Program_List"])}
        max_key = max(program_dict, key=program_dict.get)
        max_value = program_dict[max_key]

        result_dict = {key: (value / max_value) for key, value in program_dict.items() if key != max_key}
        return result_dict
    finally:
        # Cleanup: Remove the temporary file
        os.remove(temp_file_path)

def get_ir2vec_sym_obs(ir_code):
    # Save the LLVM IR code to a temporary file
    temp_file_path = save_ir_code_to_temp_file(ir_code)

    try:
        # Call i2v.generateEmbeddings with the temporary file path
        emb = i2v.generateEmbeddings(temp_file_path, "sym", "p")

        # Process the embeddings as needed
        program_dict = {index: value for index, value in enumerate(emb["Program_List"])}
        max_key = max(program_dict, key=program_dict.get)
        max_value = program_dict[max_key]

        result_dict = {key: (value / max_value) for key, value in program_dict.items() if key != max_key}
        return result_dict
    finally:
        # Cleanup: Remove the temporary file
        os.remove(temp_file_path)