# 预处理数据，剔除无效空格，按句子、段落划分纬度。

import re

input_path = "./original_text/小王子.txt"
output_path = "./tmp/小王子-clean.txt"

def split_text_into_sentences(text):
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


if __name__ == "__main__":    
    with open(input_path, "r", encoding="gbk") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip().replace(" ", "")
            if not line:
                continue
            sentences = split_text_into_sentences(line)
            for s in sentences:
                if s.strip():
                    outfile.write(s + "\n")