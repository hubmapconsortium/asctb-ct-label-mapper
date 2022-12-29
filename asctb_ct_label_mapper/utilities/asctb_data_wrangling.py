import base64, json, sys, re, os, pandas as pd
from requests import get
from pprint import pprint


# Add code to read ASCTB config-sheet from CCF-reporter repo
# Parse the google sheet to create asctb dataframe

def get_ccf_reporter_sheet_config(verbose=False):
    """Accesses the Github repo for CCF-reporter, pulls the sheet-config at below URL.
    `https://github.com/hubmapconsortium/ccf-asct-reporter/blob/main/projects/v2/src/assets/sheet-config.json`
    
    This sheet contains links for all ASCT+B organs --> dataset URL.

    Args:
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.
    
    Return:
        list(dict): Python version of the `sheet_config.json` file.
    """
    USER = 'hubmapconsortium'
    REPOSITORY_NAME = 'ccf-asct-reporter'
    FILE_PATH = '/projects/v2/src/assets/sheet-config.json'
    sheet_config = None
    response = get(url=f'https://api.github.com/repos/{USER}/{REPOSITORY_NAME}/contents/{FILE_PATH}')
    if response.status_code == 200:
        json_response = response.json()
        decoded_content = base64.b64decode(json_response['content']) # Github returns response in b64 encoding
        json_string = decoded_content.decode('utf-8')
        sheet_config = json.loads(json_string)
    else:
        print(f'Error {response.status_code}! Something went wrong while trying to read the CCF-Reporter Github repo sheet-config file...')
    if verbose:  print(f'sheet_config = \n{sheet_config}')
    return sheet_config


def get_asctb_data_url(asctb_organ='Lung', asctb_organ_version='v1.1', verbose=False):
    """Reads the sheet-config from the CCF-reporter Github repo, and parses it to fetch the dataset-link for the specific organ and version.

    Args:
        asctb_organ (str, optional): Defaults to 'Lung'.
        asctb_organ_version (str, optional): Defaults to 'v1.1'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        str: URL for the specific ASCT+B organ and version.
    """
    GOOGLE_SHEETS_BASE_URL = 'https://docs.google.com/spreadsheets/d/'
    SHEET_CONFIG = get_ccf_reporter_sheet_config(verbose=verbose)
    if not SHEET_CONFIG:
        sys.exit('Couldnt access the CCF-Reporter Github!')

    for asctb_dataset in SHEET_CONFIG:
        if asctb_dataset['name'].lower() == asctb_organ.lower():
            if verbose:  pprint(asctb_dataset)
            for version_metadata in asctb_dataset['version']:
                if version_metadata['viewValue'] == asctb_organ_version:
                    version_name = version_metadata['value']
                    google_sheets_url = GOOGLE_SHEETS_BASE_URL + version_metadata['sheetId']
                    if verbose:  print(version_name, google_sheets_url)
                    return google_sheets_url
    return None

    


def fetch_ct_info_from_asctb_google_sheet(asctb_organ='Lung', asctb_organ_version='v1.1', filepath='ontology_embeddings/', filename='ASCTB_Lungv1_2.csv', verbose=False):
    """Fetches and parses the ASCT+B dataset for the specific organ and version.
    
    Processes the ASCTB dataset in the tuples of 3 columns like `[CT/1/ID, CT/1, CT/1/LABEL], [CT/2/ID, CT/2, CT/2/LABEL], ...`
    
    Finally, produces a 3-column dataframe of `['CT_NAME', 'CT_ID', 'CT_LABEL']` with all unique CT-information for that Organ and version.

    Args:
        asctb_organ (str, optional): Defaults to 'Lung'.
        asctb_organ_version (str, optional): Defaults to 'v1.1'.
        filepath (str, optional): Target file directory. Defaults to 'ontology_embeddings/'.
        filename (str, optional): Target file name that'll contain the ASCTB NLP embeddings. Defaults to 'ASCTB_Lungv1_2.csv'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        pd.DataFrame: 3-column dataframe of `['CT_NAME', 'CT_ID', 'CT_LABEL']`.
    """

    GOOGLE_SHEET_URL = get_asctb_data_url(asctb_organ=asctb_organ, asctb_organ_version=asctb_organ_version, verbose=verbose)
    if not GOOGLE_SHEET_URL:
        sys.exit(f'Could not fetch the Google-Sheet URL for {asctb_organ} {asctb_organ_version}...')
    raw_asctb_df = pd.read_csv(f'{GOOGLE_SHEET_URL}/export?format=csv', skiprows=10)

    regex_ctid = '^CT/[0-9]*/ID$'
    regex_ctname = '^CT/[0-9]*$'
    regex_ctlabel = '^CT/[0-9]/LABEL$'
    celltype_columns = sorted([col for col in raw_asctb_df.columns if re.match(f'{regex_ctid}|{regex_ctlabel}|{regex_ctname}', col)])
    raw_asctb_df = raw_asctb_df[celltype_columns]

    asctb_df = pd.DataFrame(columns=['CT_NAME', 'CT_ID', 'CT_LABEL'])
    for i in range(0, len(celltype_columns), 3):
        next_hierarchy_df = raw_asctb_df[celltype_columns[i:i+3]]
        if verbose:  print(f'next_hierarchy_df={next_hierarchy_df.shape}')
        next_hierarchy_df.columns = asctb_df.columns
        next_hierarchy_df = next_hierarchy_df.dropna(how='all')
        if verbose:  print(f'After dropping null rows, next_hierarchy_df={next_hierarchy_df.shape}')
        asctb_df = pd.concat([ asctb_df,  next_hierarchy_df ], axis=0)

    asctb_df = asctb_df.drop_duplicates()
    asctb_df = asctb_df[['CT_ID', 'CT_NAME', 'CT_LABEL']]
    asctb_df.loc[asctb_df['CT_ID'].isna(), 'CT_ID'] = 'ASCTB CT_ID UNK'

    try:
        os.makedirs(filepath, exist_ok=True)
        asctb_df.to_csv(filepath+filename, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f'Something went wrong while trying to write the ASCT+B dataframe to a CSV file: {e}')
    return asctb_df





def get_ontobee_response(cell_ontology_id='CL_0002062', verbose=False):
    """Reusable simple function to invoke the Ontobee API and fetch metadata about a specific CT-ID.

    Args:
        cell_ontology_id (str, optional): Defaults to 'CL_0002062'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        str: Response from API call.
    """
    cell_ontology_id = cell_ontology_id.replace(':', '_')
    ONTOBEE_BASE_URL = "http://www.ebi.ac.uk/ols/api/ontologies/cl/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F"
    response = get(ONTOBEE_BASE_URL+cell_ontology_id, headers={'Accept':'application/json'}, data={})
    return response



def get_cell_ontology_label_from_id(cell_ontology_id='CL_0002062', verbose=False):
    """Adds a new column with cleaned CT_ID string, and fetches Label from the Ontobee API.

    Args:
        cell_ontology_id (str, optional): CT-ID with ':' replaced by '_'. Defaults to 'CL_0002062'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.
    """
    label = 'NaN'
    response = get_ontobee_response(cell_ontology_id)

    if response.status_code!=200:
        if verbose:  print(f'Something went wrong while using the Ontobee API for {cell_ontology_id}...')
    else:
        if verbose:  print(f'Found label for {cell_ontology_id}!')
        label = response.json()['_embedded']['terms'][0]['label']
    if verbose:  print(f'{cell_ontology_id} has label={label}')
    return label





def get_cell_ontology_definition_from_id(cell_ontology_id='CL_0002062', verbose=False):
    """Fetches Cell-Type definition of specific CT-ID from the Ontobee API.

    Args:
        cell_ontology_id (str, optional): CT-ID with ':' replaced by '_'. Defaults to 'CL_0002062'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.
    """
    definition = 'NaN'
    response = get_ontobee_response(cell_ontology_id)

    if response.status_code!=200:
        if verbose:  print(f'Something went wrong while using the Ontobee API for {cell_ontology_id}...')
    else:
        if verbose:  print(f'Found label for {cell_ontology_id}!')
        try:
            definition = response.json()['_embedded']['terms'][0]['annotation']['definition'][0]
        except Exception as e:
            if verbose:  print(f'Something went wrong while trying to access the definition of `{cell_ontology_id}`...{e}')

    if verbose:  print(f'{cell_ontology_id} has label={definition}')
    return definition




def fetch_asctb_definitions_cell_ontology(asctb_df=None, filepath='ontology_embeddings', filename='ASCTB_Lungv1_2.csv'):
    """Reads the ASCT+B source-file if input dataframe is empty, and fetches the description metadata from Cell-Ontology.
    Merge the description with current text-data in ASCT+B, so that we can generate more informative encodings.

    > Ideally this file should contain the entire ASCTB CT-ID vs CT-text information.

    Args:
        asctb_df (pd.DataFrame): Dataframe of ASCTB parsed 3 columns `['CT_NAME', 'CT_ID', 'CT_LABEL']`. Defaults to None.
        filepath (str, optional): Defaults to 'ontology_embeddings'.
        filename (str, optional): Defaults to 'ASCTB_Lungv1_2.csv'.

    Returns:
        pd.DataFrame: DataFrame of ASCT+B data merged with Cell-Ontology definitions of Cell-Types.
    """
    tgt_filepath = filepath+filename
    if asctb_df.empty or asctb_df.shape[0]==0:
        asctb_df = pd.read_csv(tgt_filepath)
    asctb_df['definition'] = asctb_df['CT_ID'].str.replace(':','_').apply(get_cell_ontology_definition_from_id)
    asctb_df['all_text'] = asctb_df['CT_NAME'] + ' ' + asctb_df['CT_LABEL'] + ' ' + asctb_df['definition']
    asctb_df.loc[asctb_df['all_text'].isna(), 'all_text'] = asctb_df.loc[asctb_df['all_text'].isna(), 'CT_NAME']
    asctb_df = asctb_df.reset_index(drop=True)

    try:
        os.makedirs(filepath, exist_ok=True)
        asctb_df.to_csv(tgt_filepath, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f'Something went wrong while trying to write the ASCT+B dataframe to a CSV file: {e}')
    return asctb_df