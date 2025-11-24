#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的 PDF 数据处理和插入 Milvus 的流水线脚本

解决 Milvus 检索返回空结果的问题
根本原因: 集合中没有数据 (num_entities = 0)

完整流程:
1. 布局分析: 从 PDF 提取图片和表格
2. 生成标题: 为图片和表格生成 caption
3. 生成 embedding: 将数据插入 Milvus
"""

import os
import sys
from pymilvus import MilvusClient

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from layout_analysis import LayoutAnalysis
from get_table_figure_caption import TableFigureMatch
from get_mm_embedding import ImageEmbedding, TextEmbedding


def run_pipeline(
    pdf_path: str,
    milvus_uri: str = "http://192.168.37.53:19530",
    query: str = None,
    run_qa: bool = False
):
    """
    运行完整的数据处理流水线,可选择运行问答
    
    Args:
        pdf_path: PDF 文件路径
        milvus_uri: Milvus 服务器地址
        query: 要问的问题(如果提供,则在数据插入后自动运行问答)
        run_qa: 是否在数据插入后运行问答
    """
    print("="*60)
    print("PDF 多模态 RAG 数据处理流水线")
    print("="*60)
    print(f"\nPDF 文件: {pdf_path}")
    print(f"Milvus URI: {milvus_uri}")
    
    # 检查 PDF 文件是否存在
    if not os.path.exists(pdf_path):
        print(f"\n❌ 错误: PDF 文件不存在: {pdf_path}")
        return
    
    # 步骤 1: 布局分析 - 提取图片和表格
    print("\n" + "="*60)
    print("步骤 1/3: 布局分析 - 提取图片和表格")
    print("="*60)
    try:
        layout_analyzer = LayoutAnalysis(pdf_file_path=pdf_path)
        print("开始布局分析...")
        layout_analyzer.run()
        print("✅ 布局分析完成")
    except Exception as e:
        print(f"❌ 布局分析失败: {e}")
        print("请检查:")
        print("  1. MinerU 及其依赖是否已正确安装 (mineru, pypdfium2 等)")
        print("  2. PDF 文件是否可读")
        return
    
    # 步骤 2: 生成标题
    print("\n" + "="*60)
    print("步骤 2/3: 为图片和表格生成标题")
    print("="*60)
    try:
        caption_generator = TableFigureMatch(pdf_file_path=pdf_path)
        print("开始生成标题...")
        caption_generator.run()
        print("✅ 标题生成完成")
    except Exception as e:
        print(f"❌ 标题生成失败: {e}")
        print("请检查:")
        print("  1. 布局分析结果文件是否存在")
        print("  2. output 目录是否有写入权限")
        return
    
    # 步骤 3: 生成 embedding 并插入 Milvus
    print("\n" + "="*60)
    print("步骤 3/3: 生成 embedding 并插入 Milvus")
    print("="*60)
    try:
        # 连接 Milvus
        print(f"连接 Milvus: {milvus_uri}")
        client = MilvusClient(uri=milvus_uri, db_name="default")
        print("✅ Milvus 连接成功")
        
        # 3a: 插入图片 embedding
        print("\n[3a] 处理图片和表格的 embedding...")
        image_embedding = ImageEmbedding(
            pdf_file_path=pdf_path, 
            milvus_client=client
        )
        image_embedding.run()
        print("✅ 图片 embedding 插入完成")
        
        # 3b: 插入文本 embedding
        print("\n[3b] 处理文本的 embedding...")
        text_embedding = TextEmbedding(
            pdf_file_path=pdf_path,
            milvus_client=client
        )
        text_embedding.run()
        print("✅ 文本 embedding 插入完成")
        
        # 验证插入结果
        print("\n" + "="*60)
        print("验证数据插入结果")
        print("="*60)
        
        image_stats = client.describe_collection("pdf_image_qa")
        text_stats = client.describe_collection("pdf_text_qa")
        
        image_count = image_stats.get('num_entities', 0)
        text_count = text_stats.get('num_entities', 0)
        
        print(f"pdf_image_qa 集合中的实体数量: {image_count}")
        print(f"pdf_text_qa 集合中的实体数量: {text_count}")
        
        if image_count > 0 and text_count > 0:
            print("\n✅ 数据插入成功!")
            print("\n现在你可以运行查询脚本:")
            print(f"  python multi-modal-rag/content_retrieval.py")
            print(f"  python multi-modal-rag/mm_doc_qa.py")
        else:
            print("\n⚠️  警告: 部分集合仍然为空")
            if image_count == 0:
                print("  - pdf_image_qa 集合为空 (可能 PDF 中没有图片/表格)")
            if text_count == 0:
                print("  - pdf_text_qa 集合为空 (可能 PDF 中没有文本)")
        
        # 步骤 4: 可选的问答功能
        if run_qa or query:
            print("\n" + "="*60)
            print("步骤 4/4: 运行多模态问答")
            print("="*60)
            
            if not query:
                query = input("\n请输入你的问题: ")
            
            print(f"\n问题: {query}")
            print("正在检索相关内容...")
            
            try:
                from content_retrieval import ContentRetrieval
                from mm_doc_qa import MultiModelQA
                
                # 检索相关内容 - 传递 pdf_path 以实现单文档隔离
                content_retriever = ContentRetrieval(
                    query=query,
                    milvus_client=client,
                    pdf_path=pdf_path  # 关键：传递 PDF 路径进行过滤
                )
                image_result, text_result = content_retriever.run()
                
                print(f"✅ 过滤条件: pdf_path == '{pdf_path}'")
                
                retrieved_text_chunks = [_['text'] for _ in text_result]
                retrieved_images = [_['image_path'] for _ in image_result]
                retrieved_captions = [_['text'] for _ in image_result]
                
                print(f"\n检索到 {len(text_result)} 个文本块和 {len(image_result)} 个图片/表格")
                print("正在生成答案...")
                
                # 运行多模态问答
                mm_qa = MultiModelQA(
                    query=query,
                    text_chunks=retrieved_text_chunks,
                    images=retrieved_images,
                    captions=retrieved_captions
                )
                answer = mm_qa.run()
                
                print("\n" + "="*60)
                print("答案:")
                print("="*60)
                print(answer)
                print("="*60)
                
            except Exception as e:
                print(f"❌ 问答过程出错: {e}")
                print("请检查:")
                print("  1. content_retrieval.py 和 mm_doc_qa.py 是否正常")
                print("  2. Ollama 服务是否正常运行")
                print("  3. 多模态 embedding 服务是否可访问")
        
        client.close()
        print("\n✅ 流水线执行完成!")
        
    except Exception as e:
        print(f"❌ Embedding 生成或插入失败: {e}")
        print("请检查:")
        print("  1. Milvus 服务是否正常运行")
        print("  2. embedding 服务器是否可访问")
        print("  3. 网络连接是否正常")
        return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="运行完整的 PDF 数据处理流水线,可选择运行问答"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="../data/LLaMA.pdf",
        help="PDF 文件路径 (默认: ../data/LLaMA.pdf)"
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default="http://192.168.37.53:19530",
        help="Milvus 服务器地址 (默认: http://192.168.37.53:19530)"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="要问的问题 (如果提供,则在数据插入后自动运行问答)"
    )
    parser.add_argument(
        "--run-qa",
        action="store_true",
        help="在数据插入后运行问答 (如果未提供 --query,则会提示输入问题)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        pdf_path=args.pdf,
        milvus_uri=args.milvus_uri,
        query=args.query,
        run_qa=args.run_qa
    )
