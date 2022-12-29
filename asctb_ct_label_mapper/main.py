import numpy as np, pandas as pd, os

from sklearn.metrics.pairwise import cosine_similarity
from asctb_ct_label_mapper.utilities.nlp_preprocessing import execute_nlp_pipeline, get_asctb_embedding
from asctb_ct_label_mapper.utilities.asctb_data_wrangling import fetch_ct_info_from_asctb_google_sheet, fetch_asctb_definitions_cell_ontology



def fetch_asctb_reference_embeddings(sentence_encoding_model, asctb_organ='Lung', asctb_version='v1.1', verbose=False):
    """Fetch ASCT+B reference embeddings from a local file, or generate them if local file doesn't exist.
    
    Currently commented out code to read pre-computed reference embeddings. Pandas not working with local file.

    Args:
        sentence_encoding_model (SentenceTransformer): Sentence encoder that transforms input cleaned sentence into 768 dimensions.
        asctb_organ (str, optional): Defaults to 'Lung'.
        asctb_version (str, optional): Defaults to 'v1.1'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        pd.DataFrame: ASCT+B data with columns `['CT_NAME', 'CT_ID', 'CT_LABEL', 'definition from Cell Ontology', '768x1 Embedding']`.
    """
    FILEPATH = 'ontology_embeddings/'
    FILENAME = f'ASCTB_{asctb_organ}{asctb_version.replace(".","_")}.pkl'
    tgt_filepath = FILEPATH+FILENAME
    if os.path.exists(tgt_filepath):
        print(f'Found pre-computed embeddings at {tgt_filepath}. Using these again!')
        asctb_embeddings_df = pd.read_pickle(tgt_filepath)
        # asctb_embeddings_df.loc[:, 'embedding_results'] = [np.fromstring(embedding).astype(np.float64) for embedding in asctb_embeddings_df['embedding_results']]
        return asctb_embeddings_df

    if verbose:  print(f'Reading the ASCT+B google sheet since local pre-computed embeddings not found at {tgt_filepath}.')
    asctb_df = fetch_ct_info_from_asctb_google_sheet(
        asctb_organ=asctb_organ, 
        asctb_organ_version=asctb_version, 
        filepath=FILEPATH, 
        filename=FILENAME, 
        verbose=verbose
    )
    if verbose:  print(f'asctb_df parsed in 3-tuple columns scheme={asctb_df.shape}')

    asctb_embeddings_df = fetch_asctb_definitions_cell_ontology(
        asctb_df=asctb_df, 
        filepath=FILEPATH, 
        filename=FILENAME
    )
    if verbose:  print(f'asctb_embeddings_df after fetching Cell Ontology definitions={asctb_embeddings_df.shape}')

    asctb_embeddings_df['embedding_results'] = asctb_embeddings_df.apply(
        get_asctb_embedding, 
        axis=1, 
        sentence_encoding_model=sentence_encoding_model,
        verbose=verbose
    )
    asctb_embeddings_df.to_csv(tgt_filepath.replace('pkl','csv'), index=False, encoding='utf-8-sig')
    asctb_embeddings_df.to_pickle(tgt_filepath)
    print(f'Created the reference embeddings for {asctb_organ}{asctb_version} at {tgt_filepath}.\nasctb_embeddings_df={asctb_embeddings_df.shape}')
    return asctb_embeddings_df






def get_best_asctb_label_match(sentence_encoding_model, asctb_embeddings_df, input_label='adventitial fibroblasts', verbose=False):
    """Performs basic NLP preprocessing to the input label, and chooses ASCT+B match based upon cosine-similarity.
    The Embeddings Dataframe needs to be precomputed with an 'embeddings_results' column containing np.ndarrays of 768x1 dimensions.

    Args:
        sentence_encoding_model (SentenceTransformer): Sentence encoder that transforms input cleaned sentence into 768 dimensions.
        asctb_embeddings_df (pd.DataFrame): Dataframe containing an 'embeddings_results' column containing precomputed np.ndarrays of 768x1 dimensions.
        input_label (str, optional): Defaults to 'adventitial fibroblasts'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        str, float, str, str: Cleaned input CT-label, and best-matching ASCT+B information found for input CT-label.
    """
    cleaned_input_label = ' '.join([execute_nlp_pipeline(word) for word in input_label.split()])
    input_embedding = sentence_encoding_model.encode(cleaned_input_label)
    reference_embeddings = np.row_stack(asctb_embeddings_df['embedding_results'])
    if verbose:  print(f'Trying to match `{input_label}` with reference_embeddings={reference_embeddings.shape}\n{reference_embeddings}')
    cosine_similarities = cosine_similarity(
        reference_embeddings , 
        input_embedding.reshape(1,-1)
    )
    if verbose:  print(f'cosine_similarities={cosine_similarities.shape}\n{cosine_similarities}')
    best_match_index = np.argmax(cosine_similarities)
    cosine_sim_match_score = np.max(cosine_similarities)
    best_match_ct_id = asctb_embeddings_df.loc[best_match_index, 'CT_ID']
    best_match_ct_label = asctb_embeddings_df.loc[best_match_index, 'CT_NAME']
    
    if verbose:  print(f'input_label={input_label}, cosine_similarities={cosine_similarities.shape}, Cosine-Similarity={np.max(cosine_similarities)}, best_match_index={best_match_index}, Match-Found=[{best_match_ct_id} , {best_match_ct_label}]')
    return cleaned_input_label, cosine_sim_match_score, best_match_ct_id, best_match_ct_label






def map_raw_labels_to_asctb(raw_labels, sentence_encoding_model, asctb_embeddings_df, source_name='ASCT+B', verbose=False):
    """Main function to map all input raw-labels to a standard controlled vocabulary maintained by `ASCT+B`.

    For more information on ASCT+B please visit the `Master Tables` at: https://hubmapconsortium.github.io/ccf-asct-reporter/
    
    Creates a detailed df containing information on cleaned-input-label, matched ASCTB label, cosine-similarity score, etc.

    Args:
        raw_labels (iterable): A list/np.array/set of raw-input labels that need to mapped against `ASCT+B`.
        sentence_encoding_model (SentenceTransformer): Sentence encoder that transforms input cleaned sentence into 768 dimensions.
        asctb_embeddings_df (pd.DataFrame): Dataframe containing an 'embeddings_results' column containing precomputed np.ndarrays of 768x1 dimensions.
        source_name (str): Source name like CellTypist/Azimuth/PopV. Defaults to `ASCT+B`.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        pd.DataFrame: Columns include cosine-similary: `match_score`, matched CT-ID: `matched_asctb_id`, matched CT-LABEL: `matched_asctb_label`, etc.
    """
    if verbose:  print(f'Standardizing {len(raw_labels)} labels...')
    tgt_folder = 'data/'
    tgt_filename = f'{source_name}Labels_to_ASCTBLabels.csv'
    tgt_filepath = f'{tgt_folder}{tgt_filename}'

    raw_to_asctb_labels_df = pd.DataFrame({
        'source':source_name, 
        'raw_input_label':raw_labels
    })
    for input_label in raw_labels:
            cleaned_input_label, cosine_sim_match_score, best_match_ct_id, best_match_ct_label = get_best_asctb_label_match(
                sentence_encoding_model, 
                asctb_embeddings_df, 
                input_label=input_label,
                verbose=verbose
            )
            raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, 'cleaned_input_label'] = cleaned_input_label
            raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, 'match_score'] = cosine_sim_match_score
            raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, 'matched_asctb_id'] = best_match_ct_id
            raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, 'matched_asctb_label'] = best_match_ct_label
    raw_to_asctb_labels_df['exact_match_found_in_asctb'] = raw_to_asctb_labels_df['cleaned_input_label'].str.lower() == raw_to_asctb_labels_df[f'matched_asctb_label'].str.lower()
    if verbose:  print(f'Mapped all labels to ASCT+B. Now writing the dataframe at {tgt_filepath}')
    try:
        os.makedirs(tgt_folder, exist_ok=True)
        raw_to_asctb_labels_df.to_csv(tgt_filepath, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f'Something went wrong while trying to write the translations to a CSV file: {e}')
    return raw_to_asctb_labels_df
