import json
import os

import datasets
from datasets.tasks import TextClassification

_CITATION = None


_DESCRIPTION = """
 Arxiv dataset for summarization.
 From paper: A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents" by A. Cohan et al.
 See: https://aclanthology.org/N18-2097.pdf 
 See: https://github.com/armancohan/long-summarization
"""
_CITATION = """\
    @inproceedings{cohan-etal-2018-discourse,
    title = "A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents",
    author = "Cohan, Arman  and
      Dernoncourt, Franck  and
      Kim, Doo Soon  and
      Bui, Trung  and
      Kim, Seokhwan  and
      Chang, Walter  and
      Goharian, Nazli",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2097",
    doi = "10.18653/v1/N18-2097",
    pages = "615--621",
    abstract = "Neural abstractive summarization models have led to promising results in summarizing relatively short documents. We propose the first model for abstractive summarization of single, longer-form documents (e.g., research papers). Our approach consists of a new hierarchical encoder that models the discourse structure of a document, and an attentive discourse-aware decoder to generate the summary. Empirical results on two large-scale datasets of scientific papers show that our model significantly outperforms state-of-the-art models.",
}
"""
_ABSTRACT = "abstract"
_ARTICLE = "article"

class ArxivSummarizationConfig(datasets.BuilderConfig):
    """BuilderConfig for ArxivSummarization."""

    def __init__(self, **kwargs):
        """BuilderConfig for ArxivSummarization.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ArxivSummarizationConfig, self).__init__(**kwargs)


class ArxivSummarizationDataset(datasets.GeneratorBasedBuilder):
    """ArxivSummarization Dataset."""
    
    _TRAIN_FILE = "train.zip"
    _VAL_FILE = "val.zip"
    _TEST_FILE = "test.zip"

    BUILDER_CONFIGS = [
        ArxivSummarizationConfig(
            name="section",
            version=datasets.Version("1.0.0"),
            description="Arxiv dataset for summarization, concatenated sections",
        ),
        ArxivSummarizationConfig(
            name="document",
            version=datasets.Version("1.0.0"),
            description="Arxiv dataset for summarization, document",
        ),
    ]

    DEFAULT_CONFIG_NAME = "section"

    def _info(self):
        # Should return a datasets.DatasetInfo object
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _ARTICLE: datasets.Value("string"),
                    _ABSTRACT: datasets.Value("string"),
                    #"id": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/armancohan/long-summarization",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        train_path = os.path.join(dl_manager.download_and_extract(self._TRAIN_FILE), "train.txt")
        val_path = os.path.join(dl_manager.download_and_extract(self._VAL_FILE), "val.txt")
        test_path = os.path.join(dl_manager.download_and_extract(self._TEST_FILE), "test.txt")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]
    
    def _generate_examples(self, filepath):
        """Generate ArxivSummarization examples."""
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)

                """
                'article_id': str,
                'abstract_text': List[str],
                'article_text': List[str],
                'section_names': List[str],
                'sections': List[List[str]]
                """
                
                if self.config.name == "document":
                    article = [d.strip() for d in data["article_text"]]
                    article = " ".join(article)
                else:
                    article = [item.strip() for sublist in data["sections"] for item in sublist]
                    article = " \n ".join(article)

                abstract = [ab.replace("<S>", "").replace("</S>", "").strip() for ab in data["abstract_text"]]
                abstract = " \n ".join(abstract)
                yield id_, {"article": article, "abstract": abstract}