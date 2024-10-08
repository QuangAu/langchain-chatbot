import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from settings import SOURCE_FILE_NAME

QUESTION_REGEX_PATTERN = r"(?:Q[:.] ).*?(?=Q[:.] |$)"
QUESTION_REGEX_PATTERN_WITH_PAGE = r"(?:\[Page \d+\]Q[:.] ).*?(?=\[Page \d+\]Q[:.] |$)"
GUIDELINE_REGEX_PATTERN = r"(?:\n \n[a-zA-Z]).*?(?=\n \n[a-zA-Z]|$)"
GUIDELINE_REGEX_PATTERN_WITH_PAGE = r"(?:\n \n\[Page \d+\][a-zA-Z]).*?(?=\n \n\[Page \d+\][a-zA-Z]|$)"


def load_pdf_file() -> list[Document]:
    """Read pdf file then return a documents list

    Returns:
        list[Document]:
    """
    loader = PyPDFLoader(f"../data/{SOURCE_FILE_NAME}")
    documents = []
    for page in loader.load():
        text = page.page_content
        documents.append((text, page.metadata["page"]))

    return documents


def process_documents(documents: list[Document]):
    """Enhance the documents list to include metadata which indicates question, answer, page number, source.

    Args:
        documents (list[Document]): Documents from pdf file.

    Returns:
        list[Document]: Enhanced documents list. This list will be inserted into vectorstore (ChromaDB) later.
    """
    merged_text = ""
    for text, page_num in documents:
        merged_text += append_page_number(text, page_num)

    guidelines_section = merged_text[:merged_text.index("Frequently Asked Questions:  ")]
    guidelines = re.findall(GUIDELINE_REGEX_PATTERN_WITH_PAGE, guidelines_section, re.DOTALL)

    qa_section = merged_text[merged_text.index("Frequently Asked Questions:  ") + len("Frequently Asked Questions:  "):]
    qa_pairs = re.findall(QUESTION_REGEX_PATTERN_WITH_PAGE, qa_section, re.DOTALL)

    qa_documents_list = convert_items_into_document_list(qa_pairs)
    guideline_documents_list = convert_items_into_document_list(guidelines)
    qa_documents_list.extend(guideline_documents_list)

    return qa_documents_list


def append_page_number(text, page_number):
    """Append page number into the text for processing later

    Args:
        text (str): original text which needs to be inserted [Page x]
        page_number (int): page number
    """
    matched_text = re.findall(QUESTION_REGEX_PATTERN, text, re.MULTILINE)
    if not matched_text:
        matched_text = re.findall(GUIDELINE_REGEX_PATTERN, text, re.MULTILINE)

    for t in matched_text:
        t = t.strip("\n \n")
        index = text.index(t)
        text = text[:index] + f"[Page {page_number + 1}]" + text[index:]

    return text


def convert_items_into_document_list(items: list[str]) -> list[Document]:
    """Convert list of raw string into documents list.

    Args:
        items (list[str]):

    Returns:
        list[Document]:
    """
    items_with_metadata = []
    for i, pair in enumerate(items, 1):
        if "[Page" in pair:
            page = re.findall(r"\[Page (\d+)\]", pair, re.DOTALL)
            current_page = int(page[0])
        pair = re.sub(r"\[Page (\d+)\]", '', pair)
        pair = pair.strip("\n").strip("\n \n")
        current_question = pair[:pair.index("\n")].strip()
        current_answer = pair[pair.index("\n"):].strip("\n")
        doc = Document(
            page_content=pair,
            metadata={"question": current_question, "answer": current_answer, "page_number": current_page, "source": SOURCE_FILE_NAME}
        )
        items_with_metadata.append(doc)
        current_question = None
        current_answer = None

    return items_with_metadata
