from __future__ import annotations
from .processor import Processor
import re
import os


class SimpleProcessor(Processor):

    def split_text_into_sentences(self, text):
        # 处理 “引号” 括起来的对话内容，整个保留
        dialog_pattern = r'“[^”]*”'
        dialogs = re.findall(dialog_pattern, text)
        placeholders = []

        # 替换对话为占位符，后面会再恢复
        def repl_dialog(m):
            placeholders.append(m.group())
            return f"<<DIALOG_{len(placeholders) - 1}>>"

        temp_text = re.sub(dialog_pattern, repl_dialog, text)

        # 用中文句号、问号、感叹号分割（避免英文句号带来噪声）
        raw_sentences = re.split(r'[。！？]', temp_text)
        sentences = []

        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            # 恢复占位符为原始对话
            for idx, dialog in enumerate(placeholders):
                sent = sent.replace(f"<<DIALOG_{idx}>>", dialog)
            sentences.append(sent)

        return sentences

    def read_sentences_from_file(self, input_path):
        """
        从文件中读取句子，按行读取并去除空行。
        """
        sentences = []
        with open(input_path, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if line:
                    sentences.append(line)
        return sentences
    """
    A simple implementation of the Processor class that processes data by returning it as is.
    If output file exists, it reads sentences from the file.
    If not, it processes the input file, splits text into sentences, and writes them to the output file.
    The input file is expected to be encoded in 'gbk', and the output file will be in 'utf-8'.
    The sentences are split by Chinese punctuation marks (。！？) and any leading or trailing whitespace
    is removed from each sentence.
    The processed sentences are returned as a list of strings.
    """
    def process(self, input_path) -> list[str]:
        output_path = input_path.replace(".txt", "-simple.txt").replace("original_text/", "tmp/")
        if os.path.exists(output_path):
            return self.read_sentences_from_file(output_path)
        
        # create tmp directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        reuslt = []
        with open(input_path, "r", encoding="gbk") as infile, open(output_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                line = line.strip().replace(" ", "")
                if not line:
                    continue
                sentences = self.split_text_into_sentences(line)
                for s in sentences:
                    if s.strip():
                        outfile.write(s + "\n")
                        reuslt.append(s.strip())
        return reuslt

    def process_text(self, text: list[str]) -> list[str]:
        """
        Process a single text input and return a list of sentences.
        This method is useful for processing text directly without file I/O.
        """
        sentences = self.split_text_into_sentences("\n".join(text))
        return [s.strip() for s in sentences if s.strip()]