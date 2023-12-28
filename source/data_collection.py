import requests
from lxml import etree
from PyPDF2 import PdfReader

import requests
from lxml import etree
import re

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
    # print(data)
    root = etree.fromstring(data)

    paper_urls = []
    dois = []
    abstracts = []
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
            abstracts.append(entry.find('{http://www.w3.org/2005/Atom}summary').text)

    return root, paper_urls, dois, abstracts


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
    _, paper_urls, dois, abstracts = get_arxiv_data(query)
    for i in range(len(paper_urls)):
        url = paper_urls[i]
        print(url)
        get_pdf_from_url(url)
        text = extract_text_from_pdf(f'data/{url.split("/")[-1]}.pdf')
        
        data_dir = 'data/'

        formatted_text, formatted_abstract = clean_text(text, abstracts[i])

        with open(f'{data_dir}/text/{url.split("/")[-1]}.txt', 'w') as f:
            f.write(formatted_text)

        with open(f'{data_dir}/abstract/{url.split("/")[-1]}.txt', 'w') as f:
            f.write(formatted_abstract)

def clean_text(text, abstract):
    """"""
    ## Cleaning Abstract
    abstract = abstract.strip()
    # Removing word splits
    formatted_abstract = re.sub(r'(-)\n(.)', r'\2', abstract)
    # Removing unnecessary line splits
    formatted_abstract = re.sub(r'(?<![?!\.])\n(.)', r' \1', formatted_abstract)

    ## Cleaning Body
    text = text.strip()
    # Removing word splits
    formatted_text = re.sub(r'(-)\n(.)', r'\2', text)
    # Removing unnecessary line splits
    formatted_text = re.sub(r'(?<![?!\.])\n(.)', r' \1', formatted_text)
    # Removing graph symbols like /s45
    formatted_text = re.sub(r'/s(\d+)', r'', formatted_text)
    
    return formatted_text, formatted_abstract

if __name__ == '__main__':
    query = 'search_query=all:electron&start=0&max_results=3'
    get_data_from_arxiv(query)