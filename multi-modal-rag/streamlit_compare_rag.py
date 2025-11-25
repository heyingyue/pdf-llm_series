#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 对比界面 - 多模态 RAG vs 文本 RAG

功能：
1. 上传 PDF 文件
2. 输入问题
3. 并排展示两种 RAG 方案的回答
4. 显示检索到的图片和表格
"""
import streamlit as st
import os
import sys
import tempfile
import shutil
import threading
from pymilvus import MilvusClient
from PIL import Image

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入多模态 RAG 模块
from content_retrieval import ContentRetrieval
from mm_doc_qa import MultiModelQA

# 导入文本 RAG 模块
from text_rag_retrieval import TextRagRetrieval
from text_rag_qa import TextRagQA

# 导入解析与向量化所需模块
from layout_analysis import LayoutAnalysis
from vlm_caption_generator import VLMCaptionGenerator
from get_mm_embedding import ImageEmbedding as MultiModalImageEmbedding
from get_mm_embedding import TextEmbedding as MultiModalTextEmbedding
from text_rag_embedding import TextRagImageEmbedding, TextRagTextEmbedding
from get_table_figure_caption import TableFigureMatch

@st.cache_resource
def get_indexing_locks():
    """
    跨会话共享的解析任务锁：
    - multimodal: 多模态 RAG 解析与向量化
    - text: 文本 RAG 解析与向量化
    保证同一时刻每种解析任务只执行一个，避免多次点击或多个网页并发导致服务崩溃；
    同时多模态和文本 RAG 可以并行运行，互不阻塞。
    """
    return {
        "multimodal": threading.Lock(),
        "text": threading.Lock(),
    }


def _has_rag_data_for_pdf(milvus_client: MilvusClient, collection_name: str, dim: int, pdf_path: str) -> bool:
    """
    使用一次轻量级 search 检查指定 PDF 是否已经在某个集合中完成向量化。
    采用与检索阶段一致的 MilvusClient.search + filter 接口，避免依赖未在项目中使用的 API。
    """
    try:
        if not milvus_client.has_collection(collection_name):
            return False

        safe_pdf_path = pdf_path.replace('"', '\\"')
        dummy_vector = [[0.0] * dim]
        res = milvus_client.search(
            collection_name=collection_name,
            data=dummy_vector,
            limit=1,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["pdf_path"],
            filter=f'pdf_path == "{safe_pdf_path}"'
        )
        return bool(res and res[0])
    except Exception as e:
        # 不影响主流程，只在后台打印日志
        print(f"[RAG status check] search failed for collection={collection_name}, pdf={pdf_path}, error={e}")
        return False


def get_rag_status(milvus_client: MilvusClient, pdf_path: str):
    """
    检查当前 PDF 在 Milvus 中的解析状态：
    - multimodal: 是否完成多模态 RAG 解析（pdf_image_qa / pdf_text_qa 中有该 pdf_path）
    - text: 是否完成文本 RAG 解析（pdf_text_rag_image / pdf_text_rag_text 中有该 pdf_path）
    """
    # 多模态 RAG：图片集合 2048 维，文本集合 4096 维
    mm_image_ready = _has_rag_data_for_pdf(milvus_client, "pdf_image_qa", 2048, pdf_path)
    mm_text_ready = _has_rag_data_for_pdf(milvus_client, "pdf_text_qa", 4096, pdf_path)

    # 文本 RAG：两类集合都是 4096 维
    text_image_ready = _has_rag_data_for_pdf(milvus_client, "pdf_text_rag_image", 4096, pdf_path)
    text_text_ready = _has_rag_data_for_pdf(milvus_client, "pdf_text_rag_text", 4096, pdf_path)

    return {
        "multimodal": mm_image_ready or mm_text_ready,
        "text": text_image_ready or text_text_ready,
    }


def run_multimodal_indexing(pdf_path: str, milvus_uri: str):
    """
    针对当前 PDF 运行多模态 RAG 的解析与向量化流水线：
    1. 布局分析 + 图片/表格抽取
    2. 生成图片/表格 caption
    3. 生成多模态 embedding 并写入 pdf_image_qa / pdf_text_qa
    """
    client = MilvusClient(uri=milvus_uri, db_name="default")
    try:
        # # 1) 布局分析
        layout_analyzer = LayoutAnalysis(pdf_file_path=pdf_path)
        layout_analyzer.run()
        
        caption_generator = TableFigureMatch(pdf_file_path=pdf_path)
        print("开始生成标题...")
        caption_generator.run()
        # 2) 使用 VLM 生成图片/表格 caption
        # caption_generator = VLMCaptionGenerator(
        #     pdf_file_path=pdf_path,
        #     ollama_url="http://192.168.37.53:11434",
        #     model="qwen3-vl:32b"
        # )
        # caption_generator.run()

        # 3) 多模态 embedding 写入 pdf_image_qa / pdf_text_qa
        image_embedding = MultiModalImageEmbedding(
            pdf_file_path=pdf_path,
            milvus_client=client
        )
        image_embedding.run()

        text_embedding = MultiModalTextEmbedding(
            pdf_file_path=pdf_path,
            milvus_client=client
        )
        text_embedding.run()
    finally:
        client.close()


def run_text_rag_indexing(pdf_path: str, milvus_uri: str):
    """
    针对当前 PDF 运行文本 RAG 的解析与向量化流水线：
    1. 布局分析 + 图片/表格抽取
    2. 生成图片/表格 caption
    3. 生成文本 RAG embedding 并写入 pdf_text_rag_image / pdf_text_rag_text
    """
    client = MilvusClient(uri=milvus_uri, db_name="default")
    try:
        # 1) 布局分析
        layout_analyzer = LayoutAnalysis(pdf_file_path=pdf_path)
        layout_analyzer.run()

        # 2) 使用 VLM 生成图片/表格 caption
        caption_generator = VLMCaptionGenerator(
            pdf_file_path=pdf_path,
            ollama_url="http://192.168.37.53:11434",
            model="qwen3-vl:32b"
        )
        caption_generator.run()

        # 3) 文本 RAG embedding 写入 pdf_text_rag_image / pdf_text_rag_text
        image_embedding = TextRagImageEmbedding(
            pdf_file_path=pdf_path,
            milvus_client=client
        )
        image_embedding.run()

        text_embedding = TextRagTextEmbedding(
            pdf_file_path=pdf_path,
            milvus_client=client
        )
        text_embedding.run()
    finally:
        client.close()


# 页面配置
st.set_page_config(
    page_title="RAG 方案对比",
    page_icon="📚",
    layout="wide"
)

# 标题
st.title("📚 RAG 方案对比：多模态 RAG vs 文本 RAG")

st.markdown("""
### 两种方案的区别：

**多模态 RAG：**
- 使用多模态 embedding（图片+文本一起）
- 使用多模态模型 qwen3-vl:30b（可以"看"图片）
- 更准确，但计算成本更高

**文本 RAG：**
- 使用纯文本 embedding（只对 caption 做 embedding）
- 使用纯文本模型 qwen3:32b（只能读 caption）
- 可能不如多模态准确
""")

# 侧边栏配置
st.sidebar.header("⚙️ 配置")

# Milvus 连接
milvus_uri = st.sidebar.text_input(
    "Milvus URI",
    value="http://localhost:19530"
)

# PDF 文件上传
st.sidebar.subheader("📄 上传 PDF 文件")
uploaded_file = st.sidebar.file_uploader(
    "选择 PDF 文件",
    type=['pdf'],
    help="上传一个 PDF 文件进行分析"
)

# 处理上传的文件
pdf_path = None
if uploaded_file is not None:
    try:
        # 创建临时目录（如果不存在）
        temp_dir = os.path.join(project_root, "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        # 保存上传的文件到临时目录
        pdf_filename = uploaded_file.name
        pdf_path = os.path.join(temp_dir, pdf_filename)

        # 写入文件
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"✅ 文件上传成功: {pdf_filename}")
        st.sidebar.info(f"📊 文件大小: {uploaded_file.size / 1024:.2f} KB")

    except Exception as e:
        st.sidebar.error(f"❌ 文件上传失败: {e}")
        pdf_path = None
else:
    st.sidebar.warning("⚠️ 请上传 PDF 文件")

# PDF 解析与向量化按钮（分别针对多模态 RAG 和文本 RAG）
if pdf_path:
    st.sidebar.subheader("🧮 PDF 解析与向量化")

    # 获取全局解析锁，保证跨会话的并发安全
    indexing_locks = get_indexing_locks()
    multimodal_lock = indexing_locks["multimodal"]
    text_lock = indexing_locks["text"]

    multimodal_running = multimodal_lock.locked()
    text_running = text_lock.locked()

    multimodal_button = st.sidebar.button(
        "🔄 解析并向量化（多模态 RAG）",
        disabled=multimodal_running,
        help="当前已有多模态解析任务在运行，请稍后再试。" if multimodal_running else "为多模态 RAG 解析并向量化当前 PDF"
    )

    text_button = st.sidebar.button(
        "🔄 解析并向量化（文本 RAG）",
        disabled=text_running,
        help="当前已有文本 RAG 解析任务在运行，请稍后再试。" if text_running else "为文本 RAG 解析并向量化当前 PDF"
    )

    if multimodal_button:
        # 使用非阻塞锁，防止同一时间多次点击导致重复解析
        if multimodal_lock.acquire(blocking=False):
            try:
                with st.spinner("正在为多模态 RAG 解析并向量化当前 PDF..."):
                    try:
                        run_multimodal_indexing(pdf_path=pdf_path, milvus_uri=milvus_uri)
                        st.sidebar.success("✅ 多模态 RAG 解析与向量化完成")
                    except Exception as e:
                        st.sidebar.error(f"❌ 多模态 RAG 解析失败: {e}")
            finally:
                multimodal_lock.release()
        else:
            st.sidebar.warning("已有多模态解析任务正在进行，请稍后重试。")

    if text_button:
        # 使用非阻塞锁，防止同一时间多次点击导致重复解析
        if text_lock.acquire(blocking=False):
            try:
                with st.spinner("正在为文本 RAG 解析并向量化当前 PDF..."):
                    try:
                        run_text_rag_indexing(pdf_path=pdf_path, milvus_uri=milvus_uri)
                        st.sidebar.success("✅ 文本 RAG 解析与向量化完成")
                    except Exception as e:
                        st.sidebar.error(f"❌ 文本 RAG 解析失败: {e}")
            finally:
                text_lock.release()
        else:
            st.sidebar.warning("已有文本 RAG 解析任务正在进行，请稍后重试。")

# 当前 PDF 的 RAG 解析状态（用于 UI 提示和对话前检查）
rag_status = None
if pdf_path:
    try:
        status_client = MilvusClient(uri=milvus_uri, db_name="default")
        rag_status = get_rag_status(status_client, pdf_path)

        multimodal_flag = "✅ 已解析" if rag_status["multimodal"] else "⚠️ 未解析"
        text_flag = "✅ 已解析" if rag_status["text"] else "⚠️ 未解析"

        st.sidebar.markdown("#### 📊 当前 RAG 解析状态")
        st.sidebar.markdown(
            f"- 多模态 RAG：{multimodal_flag}\n"
            f"- 文本 RAG：{text_flag}"
        )
    except Exception as e:
        st.sidebar.warning(f"⚠️ 无法自动检查 RAG 解析状态，请确认 Milvus 是否可用。\n\n详情: {e}")
    finally:
        try:
            status_client.close()
        except Exception:
            pass

# 主界面
st.markdown("---")

# 输入问题
query = st.text_input(
    "🔍 输入你的问题：",
    placeholder="例如：What is LLaMA-7B's zero-shot accuracy on RACE dataset?"
)

# 查询按钮
if st.button("🚀 开始查询", type="primary", disabled=not (query and pdf_path)):
    if not query:
        st.error("请输入问题")
    elif not pdf_path:
        st.error("请选择 PDF 文件")
    else:
        # 连接 Milvus，并在对话前自动检查是否已完成 RAG 解析
        try:
            client = MilvusClient(uri=milvus_uri, db_name="default")
            current_rag_status = get_rag_status(client, pdf_path)
            multimodal_ready = current_rag_status["multimodal"]
            text_ready = current_rag_status["text"]

            if not (multimodal_ready or text_ready):
                st.error("当前 PDF 尚未进行 RAG 解析，请先在左侧栏点击相应的“解析并向量化”按钮，或运行离线解析流水线后再进行对话。")
            else:
                # 创建两列布局，用于对比多模态 RAG 与文本 RAG 的回答和引用信息
                col1, col2 = st.columns(2)

                # ========== 多模态 RAG ==========
                with col1:
                    st.header("🎨 多模态 RAG")

                    if not multimodal_ready:
                        st.info("当前 PDF 尚未完成多模态 RAG 解析，请先在左侧栏点击“解析并向量化（多模态 RAG）”。")
                    else:
                        mm_text_result = []
                        mm_image_result = []

                        with st.spinner("检索中..."):
                            try:
                                mm_retriever = ContentRetrieval(
                                    query=query,
                                    milvus_client=client,
                                    pdf_path=pdf_path
                                )
                                mm_image_result, mm_text_result = mm_retriever.run()

                                mm_text_chunks = [_['text'] for _ in mm_text_result]
                                mm_images = [_['image_path'] for _ in mm_image_result]
                                mm_captions = [_['text'] for _ in mm_image_result]

                                st.success(f"✅ 检索到 {len(mm_text_result)} 个文本块和 {len(mm_image_result)} 个图片/表格")
                            except Exception as e:
                                st.error(f"❌ 检索失败: {e}")
                                mm_text_chunks, mm_images, mm_captions = [], [], []

                        if mm_text_result or mm_image_result:
                            with st.spinner("生成答案中..."):
                                try:
                                    mm_qa = MultiModelQA(
                                        query=query,
                                        text_chunks=mm_text_chunks,
                                        images=mm_images,
                                        captions=mm_captions
                                    )
                                    mm_answer = mm_qa.run()

                                    st.markdown("### 📝 答案：")
                                    st.markdown(mm_answer)

                                    # 显示图片
                                    if mm_images:
                                        st.markdown(f"### 🖼️ 相关图片/表格 ({len(mm_images)})：")
                                        for i, img_path in enumerate(mm_images, 1):
                                            if os.path.exists(img_path):
                                                    st.image(
                                                        img_path,
                                                        caption=f"{i}. {mm_captions[i-1][:50]}...",
                                                        use_container_width=True
                                                    )

                                    # 显示引用的文本片段，便于与文本 RAG 对比
                                    if mm_text_result:
                                        with st.expander("🔗 查看多模态 RAG 引用的文本片段"):
                                            for i, item in enumerate(mm_text_result, 1):
                                                page_no = item.get("page_no", "-")
                                                text_preview = item.get("text", "")[:200]
                                                st.markdown(f"**片段 {i}（第 {page_no} 页）**：{text_preview}...")
                                except Exception as e:
                                    st.error(f"❌ 生成答案失败: {e}")

                # ========== 文本 RAG ==========
                with col2:
                    st.header("📄 文本 RAG")

                    if not text_ready:
                        st.info("当前 PDF 尚未完成文本 RAG 解析，请先在左侧栏点击“解析并向量化（文本 RAG）”。")
                    else:
                        text_text_result = []
                        text_image_result = []

                        with st.spinner("检索中..."):
                            try:
                                text_retriever = TextRagRetrieval(
                                    query=query,
                                    milvus_client=client,
                                    pdf_path=pdf_path
                                )
                                text_image_result, text_text_result = text_retriever.run()

                                text_text_chunks = [_['text'] for _ in text_text_result]
                                text_images = [_['image_path'] for _ in text_image_result]
                                text_captions = [_['text'] for _ in text_image_result]

                                st.success(f"✅ 检索到 {len(text_text_result)} 个文本块和 {len(text_image_result)} 个图片/表格")
                            except Exception as e:
                                st.error(f"❌ 检索失败: {e}")
                                text_text_chunks, text_images, text_captions = [], [], []

                        if text_text_result or text_image_result:
                            with st.spinner("生成答案中..."):
                                try:
                                    text_qa = TextRagQA(
                                        query=query,
                                        text_chunks=text_text_chunks,
                                        images=text_images,
                                        captions=text_captions
                                    )
                                    text_answer = text_qa.run()

                                    st.markdown("### 📝 答案：")
                                    st.markdown(text_answer)

                                    # 显示图片
                                    if text_images:
                                        st.markdown(f"### 🖼️ 相关图片/表格 ({len(text_images)})：")
                                        for i, img_path in enumerate(text_images, 1):
                                            if os.path.exists(img_path):
                                                st.image(
                                                    img_path,
                                                    caption=f"{i}. {text_captions[i-1][:50]}...",
                                                    use_container_width=True
                                                )

                                    # 显示引用的文本片段，便于与多模态 RAG 对比
                                    if text_text_result:
                                        with st.expander("🔗 查看文本 RAG 引用的文本片段"):
                                            for i, item in enumerate(text_text_result, 1):
                                                page_no = item.get("page_no", "-")
                                                text_preview = item.get("text", "")[:200]
                                                st.markdown(f"**片段 {i}（第 {page_no} 页）**：{text_preview}...")
                                except Exception as e:
                                    st.error(f"❌ 生成答案失败: {e}")

        except Exception as e:
            st.error(f"❌ 连接 Milvus 失败: {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass

# 页脚
st.markdown("---")
st.markdown("""
### 💡 使用提示：
1. 确保 Milvus 服务正在运行
2. 确保已经运行过数据处理流水线（多模态和文本 RAG）
3. 选择 PDF 文件并输入问题
4. 点击"开始查询"按钮查看两种方案的对比结果
""")
