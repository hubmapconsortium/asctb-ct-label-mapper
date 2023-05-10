import numpy as np, pandas as pd, os

from sklearn.metrics.pairwise import cosine_similarity
from asctb_ct_label_mapper.utilities.nlp_preprocessing import execute_nlp_pipeline, get_asctb_embedding
from asctb_ct_label_mapper.utilities.asctb_data_wrangling import fetch_ct_info_from_asctb_google_sheet, fetch_asctb_definitions_cell_ontology



def fetch_asctb_reference_embeddings(sentence_encoding_model, asctb_organ='Lung', asctb_version='v1.1', max_text_length=150, verbose=False):
    """Fetch ASCT+B reference embeddings from a local file, or generate them if local file doesn't exist.
    
    Currently commented out code to read pre-computed reference embeddings. Pandas not working with local file.

    Args:
        sentence_encoding_model (SentenceTransformer): Sentence encoder that transforms input cleaned sentence into 768 dimensions.
        asctb_organ (str, optional): Defaults to 'Lung'.
        asctb_version (str, optional): Defaults to 'v1.1'.
        max_text_length (int, optional): Max amount of text to use for converting into embedding. Defaults to 150.
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
        max_text_length=max_text_length,
        verbose=verbose
    )
    asctb_embeddings_df.to_csv(tgt_filepath.replace('pkl','csv'), index=False, encoding='utf-8-sig')
    asctb_embeddings_df.to_pickle(tgt_filepath)
    print(f'Created the reference embeddings for {asctb_organ}{asctb_version} at {tgt_filepath}.\nasctb_embeddings_df={asctb_embeddings_df.shape}')
    return asctb_embeddings_df






def get_top_k_asctb_label_matches(sentence_encoding_model, asctb_embeddings_df, input_label='adventitial fibroblasts', k=1, verbose=False):
    """Performs basic NLP preprocessing to the input label, and chooses top K ASCT+B match based upon cosine-similarity.
    
    The Embeddings Dataframe needs to be precomputed with an 'embeddings_results' column containing np.ndarrays of 768x1 dimensions.

    Args:
        sentence_encoding_model (SentenceTransformer): Sentence encoder that transforms input cleaned sentence into 768 dimensions.
        asctb_embeddings_df (pd.DataFrame): Dataframe containing an 'embeddings_results' column containing precomputed np.ndarrays of 768x1 dimensions.
        input_label (str, optional): Defaults to 'adventitial fibroblasts'.
        k (int, optional): Fetch top-k matches based on Cosine-similarity for current raw-input CellType-label. Defaults to 1.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        str, list{float}, list(str), list(str): Cleaned input CT-label, and best-matching ASCT+B information found for input CT-label.
    """
    cleaned_input_label = ' '.join([execute_nlp_pipeline(word) for word in input_label.split()])
    input_embedding = sentence_encoding_model.encode(cleaned_input_label)
    reference_embeddings = np.row_stack(asctb_embeddings_df['embedding_results'])
    if verbose:  print(f'Trying to match `{input_label}` with reference_embeddings={reference_embeddings.shape}\n{reference_embeddings}')
    cosine_similarities = cosine_similarity(
        reference_embeddings , 
        input_embedding.reshape(1,-1)
    ).reshape(-1)
    if verbose:  print(f'cosine_similarities={cosine_similarities.shape}\n{cosine_similarities}')
    best_match_indexes = np.argsort(-cosine_similarities)[:k]
    cosine_sim_match_scores = cosine_similarities[best_match_indexes]
    if verbose:  print(f'best_match_indexes={best_match_indexes}, cosine_sim_match_scores={cosine_sim_match_scores}')

    best_match_ct_ids = asctb_embeddings_df.loc[best_match_indexes, 'CT_ID'].to_list()
    best_match_ct_labels = asctb_embeddings_df.loc[best_match_indexes, 'CT_NAME'].to_list()
    best_match_ct_texts = asctb_embeddings_df.loc[best_match_indexes, 'all_text'].to_list()
    if verbose:  print(f'input_label={input_label}, cosine_similarities={cosine_similarities.shape}, Top-{k} Cosine-Similarities={cosine_similarities}, best_match_index={best_match_indexes}, Match-Found=[{best_match_ct_ids} , {best_match_ct_labels}]')

    return cleaned_input_label, cosine_sim_match_scores, best_match_ct_ids, best_match_ct_labels, best_match_ct_texts





def get_exact_asctb_match_info(ct_name_cleaned, asctb_embeddings_df, src_colname='CT_ID', verbose=False):
    """Return the exactly matching information from the ASCT+B table based on cleaned CT-NAME.

    Args:
        ct_name_cleaned (str): Cleaned Cell-Type name field.
        asctb_embeddings_df (pd.DataFrame): Dataframe containing an 'embeddings_results' column containing precomputed np.ndarrays of 768x1 dimensions.
        src_colname (str, optional): Column to fetch from the ASCT+B embeddings dataframe. Defaults to 'CT_ID'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        str: ASCT+B information for CT-names found to exactly match with ASCT+B.
    """
    if verbose:  print(ct_name_cleaned)
    exact_matching_info = '1.0' if 'match_score'==src_colname else asctb_embeddings_df.loc[asctb_embeddings_df['CT_NAME_CLEANED']==ct_name_cleaned, src_colname].values[0]
    if verbose:  print(exact_matching_info)
    return exact_matching_info




def overwrite_exact_asctb_matches(raw_to_asctb_labels_df, asctb_embeddings_df, k, verbose=False):
    """Update ASCT+B information for CT-names found to exactly match with ASCT+B.

    Args:
        raw_to_asctb_labels_df (pd.DataFrame): Columns include cosine-similary: `match_score`, matched CT-ID: `matched_asctb_id`, matched CT-LABEL: `matched_asctb_label`, etc.
        asctb_embeddings_df (pd.DataFrame): Dataframe containing an 'embeddings_results' column containing precomputed np.ndarrays of 768x1 dimensions.
        k (int, optional): Fetch top-k matches based on Cosine-similarity for current raw-input CellType-label. Defaults to 1.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        pd.DataFrame: Updated dataframe with exactly matching ASCT+B information overwritten.
    """
    if verbose:  print('Cleaning up the CT-NAME field in the ASCT+B embeddings.')
    asctb_embeddings_df['CT_NAME_CLEANED'] = asctb_embeddings_df['CT_NAME'].apply(lambda input_label : ' '.join([execute_nlp_pipeline(word) for word in input_label.split()]))

    src_to_target_columns = {
        'match_score' : 'match_score_1',
        'CT_ID' : 'matched_asctb_id_1',
        'CT_NAME' : 'matched_asctb_label_1',
        'definition' : 'matched_asctb_text_1'
    }      

    for src_colname, tgt_colname in src_to_target_columns.items():
        raw_to_asctb_labels_df.loc[
            raw_to_asctb_labels_df['cleaned_input_label'].isin(asctb_embeddings_df['CT_NAME_CLEANED']), tgt_colname
        ] = raw_to_asctb_labels_df.loc[
                raw_to_asctb_labels_df['cleaned_input_label'].isin(asctb_embeddings_df['CT_NAME_CLEANED']), 'cleaned_input_label'
            ].apply(
                get_exact_asctb_match_info, 
                asctb_embeddings_df=asctb_embeddings_df, 
                src_colname=src_colname, 
                verbose=verbose
            )
    if verbose:  print('Fetched all exact information from ASCT+B !')

    for i in range(1, k):
        raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['cleaned_input_label'].isin(asctb_embeddings_df['CT_NAME_CLEANED']), f'match_score_{i+1}'] = np.NaN
        raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['cleaned_input_label'].isin(asctb_embeddings_df['CT_NAME_CLEANED']), f'matched_asctb_id_{i+1}'] = np.NaN
        raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['cleaned_input_label'].isin(asctb_embeddings_df['CT_NAME_CLEANED']), f'matched_asctb_label_{i+1}'] = np.NaN
        raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['cleaned_input_label'].isin(asctb_embeddings_df['CT_NAME_CLEANED']), f'matched_asctb_text_{i+1}'] = np.NaN
    return raw_to_asctb_labels_df




def map_raw_labels_to_asctb(raw_labels, sentence_encoding_model, asctb_embeddings_df, source_name='ASCT+B', k=1, verbose=False):
    """Main function to map all input raw-labels to a standard controlled vocabulary maintained by `ASCT+B`.

    For more information on ASCT+B please visit the `Master Tables` at: https://hubmapconsortium.github.io/ccf-asct-reporter/
    
    Creates a detailed df containing information on cleaned-input-label, matched ASCTB label, cosine-similarity score, etc.

    Args:
        raw_labels (iterable): A list/np.array/set of raw-input labels that need to mapped against `ASCT+B`.
        sentence_encoding_model (SentenceTransformer): Sentence encoder that transforms input cleaned sentence into 768 dimensions.
        asctb_embeddings_df (pd.DataFrame): Dataframe containing an 'embeddings_results' column containing precomputed np.ndarrays of 768x1 dimensions.
        source_name (str): Source name from where the raw_labels originate, ex- CellTypist/Azimuth/PopV. Defaults to `ASCT+B`.
        k (int, optional): Fetch top-k matches based on Cosine-similarity for current raw-input CellType-label. Defaults to 1.
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
            cleaned_input_label, cosine_sim_match_scores, best_match_ct_ids, best_match_ct_labels, best_match_ct_texts = get_top_k_asctb_label_matches(
                sentence_encoding_model, 
                asctb_embeddings_df, 
                input_label=input_label,
                k=k,
                verbose=verbose
            )
            raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, 'cleaned_input_label'] = cleaned_input_label
            for i in range(k):
                raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, f'match_score_{i+1}'] = cosine_sim_match_scores[i]
                raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, f'matched_asctb_id_{i+1}'] = best_match_ct_ids[i]
                raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, f'matched_asctb_label_{i+1}'] = best_match_ct_labels[i]
                raw_to_asctb_labels_df.loc[raw_to_asctb_labels_df['raw_input_label']==input_label, f'matched_asctb_text_{i+1}'] = best_match_ct_texts[i]
    
    raw_to_asctb_labels_df = overwrite_exact_asctb_matches(raw_to_asctb_labels_df, asctb_embeddings_df, k, verbose=verbose)
    
    if verbose:  print(f'Mapped all labels to ASCT+B. Now writing the dataframe at {tgt_filepath}')
    try:
        os.makedirs(tgt_folder, exist_ok=True)
        raw_to_asctb_labels_df.to_csv(tgt_filepath, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f'Something went wrong while trying to write the translations to a CSV file: {e}')
    return raw_to_asctb_labels_df
