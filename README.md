
# 相似度检索demo

## 环境要求

执行以下命令创建环境

```shell
python3 -m venv .venv
source .venv/bin/activate
```

需要安装：
1. `faiss`，用于建立索引，有cpu（`faiss-cpu`）和gpu（`faiss-gpu`）两种版本，gpu版本需有 CUDA 环境（需要 NVIDIA 显卡 + 安装好 CUDA 驱动，可能需要和你本地的 CUDA 版本匹配，否则建议用 conda 安装），这里我使用cpu版本。
2. `sentence_transformers`，用于生成向量。

```shell
pip3 install -r ./requirements.txt 
```

## 任务执行

第一步：预处理数据。执行 `pre_process_data.py`，清理空格，划分句子。文件生成在tmp文件夹。不依赖任何三方库。

第二步：计算向量并保存到本地。执行 `gen_vectors.py`。依赖 `sentence_transformers` 库。这里我们使用 [uer/sbert-base-chinese-nli](https://huggingface.co/uer/sbert-base-chinese-nli) 模型，`sentence-transformers` 会自动从 [HuggingFace Hub](https://huggingface.co) 下载模型。如想下载到本地使用，可以先自动下载，然后去本地缓存目录(~/.cache/huggingface/transformers)下寻找，并指定加载本地模型文件夹。

第三步：使用 `faiss` 实现最简单的向量检索功能平面索引：
1. 平面索引(IndexFlatL2)：执行`IndexFlatL2.py`。依赖 `faiss-cpu`。

其他：
我们这里使用的[uer/sbert-base-chinese-nli](https://huggingface.co/uer/sbert-base-chinese-nli) 模型适合用于语义相似度检索，对于提问形式的搜索并不友好，如果想加强提问形式的回答效果，可能要考虑使用其他支持问答模式的模型，如[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)、[shibing624/text2vec-base-chinese-sentence](https://huggingface.co/shibing624/text2vec-base-chinese-sentence)


## 效果
搜索"小王子伤心"（PS：这是效果最好的一个搜索了）：
```
Distances: [[226.14204 250.29906 256.32422 269.7577  281.6846 ]]
Indices: [[285 112 599 337 227]]
Index 285: Distance 226.1420440673828
sentence: 访问时间非常短，可是它却使小王子非常忧伤
---------------------
Index 112: Distance 250.29905700683594
sentence: 我在讲述这些往事时心情是很难过的
---------------------
Index 599: Distance 256.32421875
sentence: 我却很悲伤
---------------------
Index 337: Distance 269.7576904296875
sentence: “啊！”小王子大失所望
---------------------
Index 227: Distance 281.6846008300781
sentence: 他有点忧伤
---------------------
Total time taken: 3.24 seconds. Search time: 0.12 seconds
```

## 参考
[向量数据库入坑指南：聊聊来自元宇宙大厂 Meta 的相似度检索技术 Faiss](https://soulteary.com/2022/09/03/vector-database-guide-talk-about-the-similarity-retrieval-technology-from-metaverse-big-company-faiss.html)