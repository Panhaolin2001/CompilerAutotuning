import IR2Vec as i2v

def get_ir2vec_fa_obs(ir_file_path):
    emb = i2v.generateEmbeddings(ir_file_path, "fa", "p")
    program_dict = {index: value for index, value in enumerate(emb["Program_List"])}
    return program_dict

def get_ir2vec_sym_obs(ir_file_path):
    emb = i2v.generateEmbeddings(ir_file_path, "sym", "p")
    program_dict = {index: value for index, value in enumerate(emb["Program_List"])}
    return program_dict
