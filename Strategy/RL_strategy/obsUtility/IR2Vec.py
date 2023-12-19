import IR2Vec as i2v

def get_ir2vec_fa_obs(ir_file_path):
    emb = i2v.generateEmbeddings(ir_file_path, "fa", "p")
    program_dict = {index: value for index, value in enumerate(emb["Program_List"])}
    max_key = max(program_dict, key=result_dict.get)
    max_value = program_dict[max_key]

    result_dict = {key: (value / max_value) for key, value in program_dict.items() if key != max_key}
    return result_dict

def get_ir2vec_sym_obs(ir_file_path):
    emb = i2v.generateEmbeddings(ir_file_path, "sym", "p")
    program_dict = {index: value for index, value in enumerate(emb["Program_List"])}
    max_key = max(program_dict, key=result_dict.get)
    max_value = program_dict[max_key]

    result_dict = {key: (value / max_value) for key, value in program_dict.items() if key != max_key}
    return result_dict
