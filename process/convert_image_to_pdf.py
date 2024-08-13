# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: convert_image_to_pdf.py
# @time: 2024/8/5 16:31
import img2pdf
from PIL import Image
import os

# storing image path
img_path = "../data/病历.jpeg"

# storing pdf path
pdf_path = img_path.replace("jpeg", "pdf")

# opening image
image = Image.open(img_path)

# converting into chunks using img2pdf
pdf_bytes = img2pdf.convert(image.filename)

# opening or creating pdf file
file = open(pdf_path, "wb")

# writing pdf files with chunks
file.write(pdf_bytes)

# closing image file
image.close()

# closing pdf file
file.close()

# output
print("Successfully made pdf file")
