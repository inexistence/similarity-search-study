"""
faiss 中最简单的索引，便是没有使用任何花哨技巧（压缩、分区等）的平面索引：IndexFlatL2。
当我们使用这种索引的时候，我们查询的数据会和索引中所有数据进行距离计算，获取它们之间的 L2 距离（欧几里得距离）。
因为它会尽职尽责的和所有数据进行比对，所以它是所有索引类型中最慢的一种，但是也是最简单和最准确的索引类型。
同时，因为类型简单，也是内存占用量最低的类型。
而它采取的遍历式查找，也会被从业者打趣称之为“暴力搜索”。
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
from textprocessor.simple_processor import SimpleProcessor
from textprocessor.window_processor import WindowProcessor
import utils
from textprocessor.processor import Processor

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
    index_path = utils.get_index_path(model_name=model_name, filename="小王子", processor_name=processor_name)
    embeddings_path = utils.get_embeddings_path(model_name=model_name, filename="小王子", processor_name=processor_name)
    if is_file_exists(index_path):
        # 如果索引文件存在，直接读取本地保存的索引
        index = read_index(index_path)
        print(f"Index loaded from {index_path}")
    else:
        # 如果索引文件不存在，则通过向量创建索引
        print("Index file not found, creating a new index...")
        # 加载本地保存的向量
        sentence_embeddings: np.ndarray = np.load(embeddings_path)
        print(f"Loaded {sentence_embeddings.shape}")
        # 创建 IndexFlatL2 索引
        dimension = sentence_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # 创建一个 L2 索引
        index.add(sentence_embeddings)  # 添加向量到索引
        print(f"Index contains {index.ntotal} vectors")
        # 保存索引到文件
        save_index(index, index_path)
        print(f"Index saved to {index_path}")
    return index

def search_index(index: faiss.Index, search_content: str, topK: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    在索引中查询向量，返回距离和索引。
    """
    model = SentenceTransformer(model_name)
    # 将查询内容转换为向量
    search_vector = model.encode([search_content])
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
    

# 以上代码创建了一个 L2 索引，并添加了向量数据。可以通过查询向量来获取最近的向量及其距离。
# 注意：如果向量数据量很大，建议使用更高效的索引结构，如 `IndexIVFFlat` 或 `IndexHNSWFlat` 等，这些索引可以提供更快的查询速度和更低的内存占用。
# 这里使用 `IndexFlatL2` 主要是为了演示基本用法，实际应用中可能需要根据数据量和查询需求选择合适的索引类型。
# # 你可以使用 `faiss.read_index` 来加载保存的索引文件。
## ```python
# from faiss import read_index
# index = read_index("./tmp/小王子-index.index")
# ```
# # 这样就可以在后续的代码中直接使用这个索引进行查询
# # 而不需要重新创建和添加向量。
# # 这对于大规模数据集尤其有用，因为索引的创建和添加向量可能需要较长时间。
# # 通过保存和加载索引，可以大大节省时间和计算资源。
# # 另外，`faiss` 还支持多种索引类型和配置选项，可以根据具体需求进行调整。
# # 例如，可以使用 `IndexIVFFlat` 来创建倒排索引，这样可以在大规模数据集上实现更快的查询速度。
# # 具体的索引类型和配置选项可以参考 `faiss` 的官方文档和示例代码。

