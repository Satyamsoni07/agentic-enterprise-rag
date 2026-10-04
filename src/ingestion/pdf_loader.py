from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages,start=1):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": file_path,
                        "page": page_number
                    }
                }
            )

    return pages