import requests
from lxml import etree
from PyPDF2 import PdfReader

import requests
from lxml import etree

def get_arxiv_data(query):
    """
    Get data from arXiv API based on query.
    
    Parameters:
    query (str): The query string for the arXiv API.

    Returns:
    tuple: A tuple containing the XML root, list of paper URLs, and list of DOIs.
    """
    url = f'http://export.arxiv.org/api/query?{query}'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    data = response.content
    root = etree.fromstring(data)

    paper_urls = []
    dois = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        pdf_link = None
        doi_link = None
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            if 'title' in link.attrib:
                if link.attrib['title'] == 'pdf':
                    pdf_link = link.attrib['href']
                elif link.attrib['title'] == 'doi':
                    doi_link = link.attrib['href']
        
        if pdf_link and doi_link:
            paper_urls.append(pdf_link)
            dois.append(doi_link)

    return root, paper_urls, dois


def get_pdf_from_url(url):
    """Get PDF from arXiv URL."""
    response = requests.get(url)

    with open(f'data/{url.split("/")[-1]}.pdf', 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Could not read {pdf_path}. Reason: {str(e)}. Skipping.")
        return ""

def get_data_from_arxiv(query):
    """Get data from arXiv API based on query."""
    root, paper_urls = get_arxiv_data(query)
    for url in paper_urls:
        print(url)
        get_pdf_from_url(url)
        text = extract_text_from_pdf(f'data/{url.split("/")[-1]}.pdf')
        print(text) 

if __name__ == '__main__':
    query = 'search_query=all:electron&start=100&max_results=1'
    _,urls,dois = get_arxiv_data(query)

    print(urls)
    print(dois)
