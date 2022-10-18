# BioNER-Cache

The pytorch implementation for our paper [Improving biomedical named entity recognition by dynamic caching inter-sentence information](https://academic.oup.com/bioinformatics/article-abstract/38/16/3976/6618522?redirectedFrom=fulltext&login=false)

## Requirements
```bash
pip install -r requirements.txt
```

## Data preparation
The training datasets should put it in folder ./dataset/NER

For the NCBI-disease, BC5CDR-chem, BC2GM dataset, please download at https://github.com/cambridgeltl/MTL-Bioinformatics-2016.

For the NLM-chem dataset, please download at  https://www.ncbi.nlm.nih.gov/research/bionlp/Data.

## Pretrained model preparation

The pretrained model should put it in folder ./pretrained_models.

For the BioBERT model, please download at https://github.com/dmis-lab/biobert.

For the BlueBERT model, please download at https://github.com/ncbi-nlp/bluebert.

For the PubMedBERT model, please download at https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract.

For the standard BERT model, please download at https://huggingface.co/bert-base-uncased.
## Usage

1. Modify the multi_task_def.yaml accoring to the specific task.
2. File preprocessing, ```python preprocess.py ```
3. String tokenize, ```python preprocess_std.py ```
4. Model training, ```python multi_train.py```. For the details of related hyperparameters, please refer to ```opts.py```
5. The ```run_example_withcahe.txt``` and ```run_example_withoutcahe.txt``` provide running commands for model training as reference.

## Reference

https://github.com/microsoft/MT-DNN

https://github.com/ncbi-nlp/BLUE_Benchmark
## Citation

```python
@article{10.1093/bioinformatics/btac422,
    author = {Tong, Yiqi and Zhuang, Fuzhen and Zhang, Huajie and Fang, Chuyu and Zhao, Yu and Wang, Deqing and Zhu, Hengshu and Ni, Bin},
    title = "{Improving biomedical named entity recognition by dynamic caching inter-sentence information}",
    journal = {Bioinformatics},
    volume = {38},
    number = {16},
    pages = {3976-3983},
    year = {2022},
    month = {06},
    abstract = "{Biomedical Named Entity Recognition (BioNER) aims to identify biomedical domain-specific entities (e.g. gene, chemical and disease) from unstructured texts. Despite deep learning-based methods for BioNER achieving satisfactory results, there is still much room for improvement. Firstly, most existing methods use independent sentences as training units and ignore inter-sentence context, which usually leads to the labeling inconsistency problem. Secondly, previous document-level BioNER works have approved that the inter-sentence information is essential, but what information should be regarded as context remains ambiguous. Moreover, there are still few pre-training-based BioNER models that have introduced inter-sentence information. Hence, we propose a cache-based inter-sentence model called BioNER-Cache to alleviate the aforementioned problems.We propose a simple but effective dynamic caching module to capture inter-sentence information for BioNER. Specifically, the cache stores recent hidden representations constrained by predefined caching rules. And the model uses a query-and-read mechanism to retrieve similar historical records from the cache as the local context. Then, an attention-based gated network is adopted to generate context-related features with BioBERT. To dynamically update the cache, we design a scoring function and implement a multi-task approach to jointly train our model. We build a comprehensive benchmark on four biomedical datasets to evaluate the model performance fairly. Finally, extensive experiments clearly validate the superiority of our proposed BioNER-Cache compared with various state-of-the-art intra-sentence and inter-sentence baselines.Code will be available at https://github.com/zgzjdx/BioNER-Cache.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac422},
    url = {https://doi.org/10.1093/bioinformatics/btac422},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/16/3976/45300873/btac422.pdf},
}
```
