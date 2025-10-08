from collections.abc import Iterable
from typing import Dict, List

from langchain_core.documents import Document


def aggregate_by_url(
    retrieved_chunks: Iterable[Document],
) -> Dict[str, List[Document]]:
    """Group retrieved chunks by source url's"""
    urls: Dict[str, List[Document]] = {}
    for doc in retrieved_chunks:
        if (url := doc.metadata["url"]) in urls:
            urls[url].append(doc)
        else:
            urls[url] = [doc]
    return urls
