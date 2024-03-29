# ASCT+B Cell-Type Label Mapper

`asctb_ct_label_mapper` is a [package](https://test.pypi.org/project/asctb-ct-label-mapper/) to ensure controlled vocabulary for annotations of scRNA-seq datasets. The goal is to enable cross-dataset or cross-experiment comparison of data by aligning annotations to a standard reference point.

Given a specific organ's scRNA-seq annotated dataset (.h5ad/.rds), you can create a translation file for mapping raw-labels to the ASCT+B naming convention.

------------------------

## General flow:

1. Create the reference-embeddings by fetching the corresponding ASCT+B organ (with latest version):

* Fetch the ASCT+B dataset from the [ASCT+B Master Tables](https://hubmapconsortium.github.io/ccf-asct-reporter/).
* Parse the data to create wrangled 3 columns `CT-ID`, `CT-Name`, `CT-Label`.
* Fetch `Description` of each unique `CT-ID` from [Cell Ontology](https://www.ebi.ac.uk/ols/ontologies/cl).
* Use NLP-preprocessing best practices for the text fields.
* Use a `Sentence-Transformer` model hosted on [Hugging Face](https://www.sbert.net/docs/pretrained_models.html) to create embeddings of shape `cx768` (`c` is the Number of unique CTs in the ASCT+B Master table).

2. For each input raw Cell-Type annotation/cluster label, create the embedding and compare it against the embeddings generated in step #1.

3. Identify the best matching ASCT+B label for the input raw label.

4. You can also visualize the agreeability of cross-dataset annotations before and after using [ASCTB CT Label Mapper](https://github.com/hubmapconsortium/asctb-ct-label-mapper).

------------------------

A walkthrough is available on Google Colab [here](https://colab.research.google.com/drive/1BNnjTheQS1x5HCfK20MnV9otp-QYQJbW?usp=sharing).

------------------------

## Architecture:

------------------------

### Step 1: Create Reference Embeddings

![Step 1: Create Reference Embeddings](/documentation/Step1.PNG?raw=True)

------------------------

### Step 2: Map input Cell-Type labels to these Reference Embeddings

![Step 2: Map input labels to Reference Embeddings](/documentation/Step2.PNG?raw=True)


------------------------

### Output: Top-2 matches from ASCT+B as suggestions for each of query Cell-Type annotation label

> Expert provides feedback in order to finalize the translation from query annotation label to ASCT+B annotation label.

![Output_summary](https://user-images.githubusercontent.com/32123572/231326852-4981234d-1f29-4f39-b1c3-a019c5065196.PNG)


------------------------

### Cosine Similarity

![Cosine Similarity](/documentation/Cosine_similarities_CTNames.PNG?raw=True)

------------------------


