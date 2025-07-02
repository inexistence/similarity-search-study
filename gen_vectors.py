from sentence_transformers import SentenceTransformer
import numpy as np

model_name = "uer/sbert-base-chinese-nli"
input_path = "./tmp/小王子-clean.txt"
output_path = f"./tmp/小王子-embeddings-{model_name.replace("/", "_")}.npy"

if __name__ == "__main__":
    model = SentenceTransformer(model_name)

    # 读取文本文件，按行分割成句子
    # 注意：这里假设每行是一个句子，实际情况可能需要根据具体文本格式调整
    # 例如：如果每行是一个段落，可以先将段落按句子分割
    sentences = []
    for line in open(input_path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sentences.append(line)

    sentence_embeddings = model.encode(sentences)
    
    print(sentence_embeddings.shape)  # (num_sentences, embedding_dim)

    # 保存向量到文件
    np.save(output_path, sentence_embeddings)