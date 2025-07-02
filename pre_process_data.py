# 预处理数据，剔除无效空格，按句子、段落划分纬度。

from textprocessor.simple_processor import SimpleProcessor
from textprocessor.window_processor import WindowProcessor

input_path = "./original_text/小王子.txt"

if __name__ == "__main__":
    # 处理文本，生成简单的句子列表。按句号、感叹号、问号和换行符分割
    SimpleProcessor().process(input_path)
    print("SimpleProcessor completed.")
    # 使用窗口处理器处理文本，生成滑动窗口的文本块。每个窗口3句，步长2句
    WindowProcessor().process(input_path)
    print("WindowProcessor completed.")