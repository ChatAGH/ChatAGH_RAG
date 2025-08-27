
def aggregate_by_url(retrieved_chunks):
    """Group retrieved chunks by source url's"""
    urls = {}
    for doc in retrieved_chunks:
        if (url := doc.metadata["url"]) in urls:
            urls[url].append(doc)
        else:
            urls[url] = [doc]
    return urls
