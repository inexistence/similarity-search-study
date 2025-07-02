def get_embeddings_path(filename: str, model_name: str, processor_name: str) -> str:
    """
    根据输入路径和模型名称生成 embeddings 的保存路径。
    """
    return f"./tmp/{filename}-embeddings-{model_name.replace('/', '_')}-{processor_name}.npy"

def get_index_path(filename: str, model_name: str, processor_name: str) -> str:
    """
    根据输入路径和模型名称生成索引的保存路径。
    """
    return f"./tmp/{filename}-index-{model_name.replace('/', '_')}-{processor_name}.index"