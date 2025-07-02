from sentence_transformers import SentenceTransformer
import numpy as np
from textprocessor.simple_processor import SimpleProcessor
from textprocessor.window_processor import WindowProcessor
import utils
from textprocessor.processor import Processor

model_name = "uer/sbert-base-chinese-nli"
input_path = "./original_text/小王子.txt"

def gen_vectors(input_path: str, processor: Processor):
    model = SentenceTransformer(model_name)

    # 获取预处理的句子
    sentences = processor.process(input_path)
    # 将预处理的句子向量化
    sentence_embeddings = model.encode(sentences)
    
    print(f"Generated embeddings for {processor.name()}.")
    print(sentence_embeddings.shape)  # (num_sentences, embedding_dim)

    # 保存向量到文件
    np.save(utils.get_embeddings_path("小王子", model_name, processor.name()), sentence_embeddings)


if __name__ == "__main__":
    gen_vectors(input_path, WindowProcessor())
    print("Vector generation completed using WindowProcessor.")

    gen_vectors(input_path, SimpleProcessor())
    print("Vector generation completed using SimpleProcessor.")

