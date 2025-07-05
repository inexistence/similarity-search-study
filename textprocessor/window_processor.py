from __future__ import annotations
from .processor import Processor
import re
import json
import os

class WindowProcessor(Processor):
    """
    A class to handle text processing with sliding windows.
    """

    def process(self, input_path) -> list[str]:
        """
        Process the input data using sliding windows to create text chunks.
        :param input_path: 输入文件路径
        :param output_path: 输出文件路径
        :param window_size: 窗口大小，默认3句
        :param stride: 步长，默认2句
        :return: 处理后的文本块列表
        """
        output_path = input_path.replace(".txt", "-_chunks.jsonl").replace("original_text/", "tmp/")
        if os.path.exists(output_path):
            return self.read_chunks_from_file(output_path)

        # create tmp directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.preprocess_text_file(input_path, output_path, window_size = 3, stride = 2)
        return self.read_chunks_from_file(output_path)
    
    def read_chunks_from_file(self, output_path):
        """
        从输出文件中读取处理后的文本块。
        :param output_path: 输出文件路径
        :return: 文本块列表
        """
        chunks = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    # 简单搞，只返回text内容
                    chunks.append(json.loads(line.strip())['text'])
        return chunks
        
    def split_sentences(self, text):
        """
        更智能的中文断句函数，支持对话保护，断句更精准
        """
        text = text.strip()
        if not text:
            return []

        # 去除多余空白
        text = re.sub(r'\s+', '', text)

        # 引号保护（同上）
        quote_blocks = {}
        def protect_quotes(m):
            key = f"__QUOTE_{len(quote_blocks)}__"
            quote_blocks[key] = m.group(0)
            return key

        # 支持中文“”和英文双引号
        text = re.sub(r'[“|"](.*?)[”|"]', protect_quotes, text)

        # 中文断句：在标点后且后面是非标点字符时加换行
        # 注意：不能断在句号后紧跟标点或引号的地方
        text = re.sub(r'(?<=[。！？])(?=[^”’」）】！？。，；：、])', '\n', text)

        # 恢复引号
        for key, value in quote_blocks.items():
            text = text.replace(key, value)

        # 切分后清理
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences

    def build_windows(self, sentences, window_size=3, stride=2):
        """
        滑动窗口拼接句子，生成检索单元
        :param sentences: 列表，包含分割后的句子
        :param window_size: 窗口大小，默认3句
        :param stride: 步长，默认2句
        """
        chunks = []
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = " ".join(sentences[i:i + window_size])
            chunks.append(chunk)
        return chunks

    def preprocess_text_file(self, input_file, output_file, window_size=3, stride=2):
        with open(input_file, "r", encoding="gbk") as f:
            raw_lines = f.readlines()

        all_chunks = []
        for idx, raw in enumerate(raw_lines):
            raw = raw.strip()
            if not raw:
                continue
            sentences = self.split_sentences(raw)
            chunks = self.build_windows(sentences, window_size, stride)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "para_id": idx, # 原始段落编号
                    "chunk_id": i # 在当前段落中的编号
                })

        with open(output_file, "w", encoding="utf-8") as f:
            for item in all_chunks:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ 处理完成，共生成 {len(all_chunks)} 条检索单元 → {output_file}")