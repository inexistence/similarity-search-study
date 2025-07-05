"""
对索引数据进行分区优化，将数据通过“沃罗诺伊图单元”（也被叫做泰森多边形）来进行切割（类似传统数据库分库分表）。
当我们想要进行针对某个数据的向量相似度检索时，会先针对向量数据和“沃罗诺伊图”的质心进行计算，求出它们之间的距离。然后，将我们的搜索范围限定在它和这个质心距离覆盖的单元内。
因为我们缩小了搜索范围，并没有像平面索引一样，进行全量的“暴力搜索”，所以我们将得到一个近似精确的答案，以及相对更快地得到数据的查询结果。
通常情况，我们会通过增加 index.nprobe 数值，来告诉 faiss 搜索更多的“格子”，以便寻找到更合适的结果。
我们需要根据业务的实际情况，来微调 index.nprobe 到合理的数值
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
    index_path = utils.get_index_path(model_name=model_name, filename="小王子", processor_name=processor_name, mode="IndexIVFlatL2")
    embeddings_path = utils.get_embeddings_path(model_name=model_name, filename="小王子", processor_name=processor_name)
    if is_file_exists(index_path):
        # 如果索引文件存在，直接读取本地保存的索引
        index = read_index(index_path)
        print(f"Index loaded from {index_path} size={str(os.path.getsize(index_path)/1024/1024)}MB")
    else:
        # 如果索引文件不存在，则通过向量创建索引
        print("Index file not found, creating a new index...")
        # 加载本地保存的向量
        sentence_embeddings: np.ndarray = np.load(embeddings_path)
        print(f"Loaded {sentence_embeddings.shape}")
        # 创建 IndexFlatL2 索引
        dimension = sentence_embeddings.shape[1] # 索引数据维度
        quantizer = faiss.IndexFlatL2(dimension)  # 分区索引的质心位置
        nlist = 50 # 要划分出多少个数据分区
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist) # 创建分区索引
        index.train(sentence_embeddings) # 添加数据之前要进行训练（https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index）
        index.add(sentence_embeddings)  # 添加向量到索引
        print(f"Index contains {index.ntotal} vectors")
        # 保存索引到文件
        save_index(index, index_path)
        print(f"Index saved to {index_path} size={str(os.path.getsize(index_path)/1024/1024)}MB")
    return index

def search_index(index: faiss.Index, search_content: str, topK: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    在索引中查询向量，返回距离和索引。
    """
    model = SentenceTransformer(model_name)
    # 将查询内容转换为向量
    search_vector = model.encode([search_content])
    # 指定搜索时需要检查的最近邻聚类中心数量​（即 Voronoi 单元数量）
    # 取值为 1 到 索引构建时的 nlist 参数值（默认为 1）。
    # 增大 nprobe​：搜索更精确（检查更多聚类区域），但速度更慢。
    # ​减小 nprobe​：搜索更快，但可能漏掉部分相似向量。
    index.nprobe = 1
    # 在索引中搜索最相似的向量
    distances, indices = index.search(search_vector, topK)
    return distances, indices

def search_with_processor(processor: Processor):
    start_time = time.time()
    # 生成索引
    index = get_index(processor.name())
    search_start_time = time.time()
    topK = 5
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
    print(f"Total time taken: {time.time() - start_time:.2f} seconds. Search time: {time.time() - search_start_time:.2f} seconds")

if __name__ == "__main__":
    print("========Using SimpleProcessor========")
    search_with_processor(SimpleProcessor())
    print("=====================================\n\n")
    print("========Using WindowProcessor========")
    search_with_processor(WindowProcessor())
    print("=====================================")    
    
