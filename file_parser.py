"""文件解析器模块：支持 TXT、PDF、DOCX、MD 格式"""
import chardet


def parse_file(uploaded_file):
    """统一文件解析入口，根据文件扩展名分发到对应解析器
    
    Args:
        uploaded_file: Streamlit 的 UploadedFile 对象
    
    Returns:
        (text, filename) 元组
    """
    filename = uploaded_file.name
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    parsers = {
        "txt": parse_txt,
        "md": parse_txt,      # Markdown 按纯文本处理
        "pdf": parse_pdf,
        "docx": parse_docx,
    }

    parser = parsers.get(suffix)
    if parser is None:
        raise ValueError(f"不支持的文件格式: .{suffix}")

    text = parser(uploaded_file)
    return text, filename


def parse_txt(uploaded_file):
    """解析 TXT/MD 文件，自动检测编码"""
    raw_bytes = uploaded_file.getvalue()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    # gb2312/gbk 是 gb18030 的子集，升级为 gb18030 兼容更多字符
    if encoding.lower().replace("-", "") in ("gb2312", "gbk", "gb18030"):
        encoding = "gb18030"
    return raw_bytes.decode(encoding, errors="replace")


def parse_pdf(uploaded_file):
    """解析 PDF 文件，提取所有页面的文本"""
    from pypdf import PdfReader
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    if not pages:
        raise ValueError("PDF 文件中未提取到任何文本内容")
    return "\n\n".join(pages)


def parse_docx(uploaded_file):
    """解析 Word (.docx) 文件，提取所有段落文本"""
    from docx import Document
    doc = Document(uploaded_file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("Word 文件中未提取到任何文本内容")
    return "\n\n".join(paragraphs)
