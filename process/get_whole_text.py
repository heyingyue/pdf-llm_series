# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: get_whole_text.py
# @time: 2024/7/24 10:54
# 使用fitz获取PDF文档中的所有文本内容
import time
import fitz

s_time = time.time()
doc = fitz.open('../data/LLaMA.pdf')

content = ""
for i in range(doc.page_count):
    page = doc[i]
    # 获取文本区域的文字
    page_content = page.get_text("blocks")
    # print(page_content)
    for record in page_content:
        if not record[-1]:
            content += record[4]
doc.close()

print(f"whole text: \n{content}")
print(f"cost time: {time.time() - s_time}")
