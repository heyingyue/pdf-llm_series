# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: is_scanned_pdf_v2.py
# @time: 2024/8/15 10:50
# 借助OCR技术，对PDF文件在OCR前后的字符串进行统计，如果 OCR前字符数/OCR后字符数 不在一定范围内（比如[0.5, 2]），则可判断为扫描版PDF。
import os
import time
import traceback

import fitz
import requests
from PIL import Image


# 使用fitz模块提取文本， 未使用OCR
def get_pdf_file_text(
        pdf_file_path: str,
        pdf_page_count: int
) -> str:
    doc = fitz.open(pdf_file_path)
    whole_text_list = []
    for i in range(pdf_page_count):
        if i < doc.page_count:
            page = doc[i]
            page_content = page.get_text("blocks")
            for record in page_content:
                if not record[-1]:
                    whole_text_list.append(record[4])
    doc.close()
    return ''.join(whole_text_list)


# 将PDF文件转换为图片
def convert_pdf_2_img(
        pdf_file: str,
        pages: int
) -> list[str]:
    """
    convert pdf to image
    :param pdf_file: pdf file path
    :param pages: convert pages number(at most)
    :return: output of image file path list
    """
    pdf_document = fitz.open(pdf_file)
    output_image_file_path_list = []
    # Iterate through each page and convert to an image
    for page_number in range(pages):
        if page_number < pdf_document.page_count:
            # Get the page
            page = pdf_document[page_number]
            # Convert the page to an image
            pix = page.get_pixmap()
            # Create a Pillow Image object from the pixmap
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Save the image
            pdf_file_name = pdf_file.split("/")[-1].split(".")[0]
            if not os.path.exists(f"../output/{pdf_file_name}"):
                os.makedirs(f"../output/{pdf_file_name}")
            save_image_path = f"../output/{pdf_file_name}/{page_number + 1}.png"
            image.save(save_image_path)
            output_image_file_path_list.append(save_image_path)
    # Close the PDF file
    pdf_document.close()
    return output_image_file_path_list


# 使用OCR技术从图片中提取文本
def get_file_content(file_path: str):
    with open(file_path, 'rb') as fp:
        return fp.read()


class CommonOcr(object):
    _app_id = '0591274538f652b47dca360c44a3cca4'
    _secret_code = 'a46398f18233f2fec5c7902b0aedb681'

    @classmethod
    def recognize(cls, img_path: str):
        try:
            # 通用文字识别
            url = 'https://api.textin.com/ai/service/v2/recognize/multipage'
            head = {'x-ti-app-id': cls._app_id, 'x-ti-secret-code': cls._secret_code}
            image = get_file_content(img_path)
            result = requests.post(url, data=image, headers=head)
            text_list = []
            if result.status_code == 200:
                for record in result.json()['result']['pages'][0]['lines']:
                    text_list.append(record['text'])
            return '\n'.join(text_list)
        except Exception as e:
            print(f"error in recognize: {traceback.format_exc()}")
            return ''


def get_pdf_images_text(
        image_path_list: list[str]
) -> str:
    image_text_list = []
    for i, image_path in enumerate(image_path_list):
        print("processing image: ", i + 1, " / ", len(image_path_list), " ...")
        image_text_list.append(CommonOcr.recognize(image_path))
    return '\n'.join(image_text_list)


if __name__ == '__main__':
    s_time = time.time()

    PDF_FILE_PATH = "../data/外国电影史.pdf"
    PDF_PAGE_COUNT = 10

    # 未使用OCR
    original_text = get_pdf_file_text(PDF_FILE_PATH, PDF_PAGE_COUNT)
    print(f"未使用OCR，{PDF_FILE_PATH}的前{PDF_PAGE_COUNT}页的字符: {original_text}")
    # 使用OCR
    output_image_path_list = convert_pdf_2_img(PDF_FILE_PATH, PDF_PAGE_COUNT)
    ocr_image_text = get_pdf_images_text(output_image_path_list)
    print(f"使用OCR，{PDF_FILE_PATH}的前{PDF_PAGE_COUNT}页的字符: {ocr_image_text}")

    # 判断是否为扫描版PDF
    print(f"未使用OCR前，{PDF_FILE_PATH}的前{PDF_PAGE_COUNT}页的字符总数: {len(original_text)}")
    print(f"使用OCR后，{PDF_FILE_PATH}的前{PDF_PAGE_COUNT}页的字符总数: {len(ocr_image_text)}")
    print(f"OCR前字符数/OCR后字符数: {len(original_text) / len(ocr_image_text)}")

    ratio = len(original_text) / len(ocr_image_text) if len(ocr_image_text) else 0
    if 0.5 < ratio < 2:
        print(f"{PDF_FILE_PATH}不是扫描版PDF")
    else:
        print(f"{PDF_FILE_PATH}是扫描版PDF")

    print(f"cost time: {time.time() - s_time}")
