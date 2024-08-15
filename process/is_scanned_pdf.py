# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: is_scanned_pdf.py
# @time: 2024/3/8 10:06
# 判断是否为扫描版pdf, 考虑文本区域占整个页面的比例, 如果小于某个阈值(比如0.05), 则认为是扫描版pdf
import time
import fitz

s_time = time.time()
doc = fitz.open('../data/外国电影史.pdf')

total_area = 0
text_area = 0
for i in range(doc.page_count):
    page = doc[i]
    # 获取page的宽和高
    total_area += page.rect[2] * page.rect[3]
    # 获取文本区域的面积
    page_content = page.get_text("blocks")
    for record in page_content:
        if not record[-1]:
            text_area += (record[3] - record[1]) * (record[2] - record[0])

doc.close()

print(text_area / total_area)
print(f"cost time: {time.time() - s_time}")

