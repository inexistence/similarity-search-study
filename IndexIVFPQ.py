"""
适用于海量数据以及数据快速增长的场景，能加速检索过程，同时减少内存空间占用。本例中数据量过少，无法体现。
乘积量化索引具备一定的压缩向量数据的功能。除了划分细粒度的“工作区域”外，还预先计算了不同向量数据之间的“距离”，让每一堆向量数据都是距离这堆向量数据的质心更近的数据。
但相比较平面索引和分区索引，faiss.IndexIVFPQ 的索引的结果精确度不高
"""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
from textprocessor.simple_processor import SimpleProcessor
from textprocessor.window_processor import WindowProcessor
import utils
from textprocessor.processor import Processor
import os

model_name = "uer/sbert-base-chinese-nli"

input_path = "./original_text/小王子.txt"

# 查询内容
search_content = "小王子伤心"

def is_file_exists(file_path: str) -> bool:
    """
    检查文件是否存在。
    """
    import os
    return os.path.exists(file_path)

def save_index(index, index_path: str):
    """
    将 FAISS 索引保存到指定路径。
    """
    faiss.write_index(index, index_path)

def read_index(index_path: str):
    """
    从指定路径读取 FAISS 索引。
    """
    return faiss.read_index(index_path)

def get_index(processor_name: str = "WindowProcessor") -> faiss.Index:
    """
    获取 FAISS 索引，如果索引文件存在则加载，否则创建新的索引。
    """
    index_path = utils.get_index_path(model_name=model_name, filename="小王子", processor_name=processor_name, mode="IndexIVFPQ")
    embeddings_path = utils.get_embeddings_path(model_name=model_name, filename="小王子", processor_name=processor_name)
    if is_file_exists(index_path):
        # 如果索引文件存在，直接读取本地保存的索引
        index = read_index(index_path)
        print(f"Index loaded from {index_path} size={str(os.path.getsize(index_path)/1024)}KB")
    else:
        # 如果索引文件不存在，则通过向量创建索引
        print("Index file not found, creating a new index...")
        # 加载本地保存的向量
        sentence_embeddings: np.ndarray = np.load(embeddings_path)
        print(f"Loaded {sentence_embeddings.shape}")
        # 创建 IndexFlatL2 索引
        dimension = sentence_embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = 50
        m = 8
        nbits_per_idx = 8
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits_per_idx) 
        index.train(sentence_embeddings)
        index.add(sentence_embeddings)  # 添加向量到索引
        print(f"Index contains {index.ntotal} vectors")
        # 保存索引到文件
        save_index(index, index_path)
        print(f"Index saved to {index_path} size={str(os.path.getsize(index_path)/1024)}KB")
    return index

def search_index(index: faiss.Index, search_content: str, topK: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    在索引中查询向量，返回距离和索引。
    """
    model = SentenceTransformer(model_name)
    # 将查询内容转换为向量
    encode_start = time.time()
    search_vector = model.encode([search_content])
    encode_cost = time.time() - encode_start
    search_start = time.time()
    # 在索引中搜索最相似的向量
    distances, indices = index.search(search_vector, topK)
    print(f"----------------")
    print(f"encode_cost={encode_cost*1000}ms search_cost={(time.time() - search_start)*1000}ms")
    print(f"----------------")
    return distances, indices

def search_with_processor(processor: Processor):
    start_time = time.time()
    # 生成索引
    index = get_index(processor.name())
    search_start_time = time.time()
    topK = 5
    index.nprobe = 1
    # 查询向量
    D, I = search_index(index, search_content, topK)
    print("Distances:", D)
    print("Indices:", I)
    sentences = processor.process(input_path)
    # 输出查询结果
    for i in range(len(I[0])):
        # D[i] 是距离，越接近越相似，I[i] 是索引
        print(f"Index {I[0][i]}: Distance {D[0][i]}")
        print(f"sentence: {sentences[I[0][i]]}")
        print("---------------------")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    print("========Using SimpleProcessor========")
    search_with_processor(SimpleProcessor())
    print("=====================================\n\n")
    print("========Using WindowProcessor========")
    search_with_processor(WindowProcessor())
    print("=====================================")    
    
