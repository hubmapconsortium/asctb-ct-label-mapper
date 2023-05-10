import matplotlib.pyplot as plt, plotly.express as px
import pandas as pd, numpy as np

from umap import UMAP
from collections import Counter
from matplotlib_venn import venn2, venn3


def make_venn_diagram(A, B, C=None, labels=['A','B','C'], title='', savefig=True):
    """General function to create a two-way or three-way Venn diagram.

    Args:
        A (set): Set of elements
        B (set): Set of elements
        C (set, optional): Set of elements. Defaults to None.
        labels (list, optional): Names/labels for A, B, C. Defaults to ['A','B','C'].
        title (str): Defaults to ''.
        savefig (bool): If True stores the venn-diagram plot to `data/threeway_intersection.png` Defaults to True.
    """
    if not C:
        C = set()
    AB_overlap = A & B  #compute intersection of set A & set B
    AC_overlap = A & C
    BC_overlap = B & C
    ABC_overlap = A & B & C
    A_rest = A - AB_overlap - AC_overlap #see left graphic
    B_rest = B - AB_overlap - BC_overlap
    C_rest = C - AC_overlap - BC_overlap
    AB_only = AB_overlap - ABC_overlap   #see right graphic
    AC_only = AC_overlap - ABC_overlap
    BC_only = BC_overlap - ABC_overlap


    plt.figure(figsize=(7,7)) 
    ax = plt.gca() 
    ax.set_title(title, color='orange', fontsize=20)
    sets = Counter()               #set order A, B, C   


    colors = ['darkviolet','deepskyblue','blue']
    if C==set():
        sets['10'] = len(A_rest)      #10 denotes A on, B off
        sets['01'] = len(B_rest)      #01 denotes A off, B on
        sets['11'] = len(AB_only)     #11 denotes A on, B on
        v = venn2(subsets=sets, set_labels=labels[:2], ax=ax, set_colors=colors[:2], alpha=0.7)
    else:
        sets['100'] = len(A_rest)      #100 denotes A on, B off, C off 
        sets['010'] = len(B_rest)      #010 denotes A off, B on, C off
        sets['001'] = len(C_rest)      #001 denotes A off, B off, C on 
        sets['110'] = len(AB_only)     #110 denotes A on, B on, C off
        sets['101'] = len(AC_only)     #101 denotes A on, B off, C on 
        sets['011'] = len(BC_only)     #011 denotes A off, B on, C on 
        sets['111'] = len(ABC_overlap) #011 denotes A on, B on, C on
        v = venn3(subsets=sets, set_labels=labels, ax=ax, set_colors=colors, alpha=0.7)
    for i, text in enumerate(v.set_labels):
        text.set_color(colors[i])

    for text in v.subset_labels:
        if not text:
            continue
        text.set_color('white')
        text.set_fontsize(16)
        text.set_fontweight('bold')
    
    if savefig:
        plt.savefig(f'data/threeway_intersection{"_"+title if title else ""}.png')
    plt.show()




def visualize_embeddings(embeddings_df, embeddings_folder='ontology_embeddings', asctb_organ='Lung', asctb_version='v1_2', umap_dimensionality=2):
    """Visualizes the reference-embeddings generated in step 1 of this package.
    Imputes the `CT_LABEL` and `definition` columns.

    Args:
        embeddings_df (pd.DataFrame): Dataframe containing an 'embeddings_results' column containing precomputed np.ndarrays of 768x1 dimensions.
        embeddings_folder (str, optional): Folder-name for the embeddings pickle file. Defaults to 'ontology_embeddings'.
        asctb_organ (str, optional): Defaults to 'Lung'.
        asctb_version (str, optional): Defaults to 'v1_2'.
        umap_dimensionality (int, optional): Defaults to 2.
    """
    embeddings_filename = f'ASCTB_{asctb_organ}{asctb_version}.pkl'
    try:
        embeddings_df = pd.read_pickle(embeddings_folder+'/'+embeddings_filename)
    except Exception as e:
        print(f'Something went wrong while trying to read the pickle file. Are you sure it exists?\n{e}')

    embeddings_df.loc[embeddings_df['CT_LABEL'].isna(), 'CT_LABEL'] = embeddings_df.loc[embeddings_df['CT_LABEL'].isna(), 'CT_NAME'].fillna('Unknown CT-Label')
    embeddings_df.loc[embeddings_df['definition'] == 'NaN', 'definition'] = embeddings_df.loc[embeddings_df['definition'] == 'NaN', 'CT_LABEL']
    embedding_matrix = embeddings_df['embedding_results'].to_numpy()

    umap_embedding = UMAP(random_state=0, n_components=umap_dimensionality)

    features = np.vstack(embedding_matrix)
    projections = umap_embedding.fit_transform(features)
    projections_df = pd.DataFrame(projections)

    N_CHARS_DEFINITION_HOVER = 150
    projections_df['Definition'] = [ definition[:N_CHARS_DEFINITION_HOVER] for definition in embeddings_df.loc[:, 'definition'].values ]
    projections_df['CT_ID'] = embeddings_df.loc[:, 'CT_ID'].replace('ASCTB CT_ID UNK', 'Unknown CT-ID').values
    projections_df['CT_LABEL'] = embeddings_df.loc[:, 'CT_LABEL']
    projections_df['CT_NAME'] = embeddings_df.loc[:, 'CT_NAME']


    fig = px.scatter(
        projections_df, 
        height=1000,
        width=1500,
        x=0, 
        y=1,
        # z=2,
        color='CT_LABEL',
        hover_name='CT_ID',
        hover_data=['CT_NAME', 'Definition'],
        title=f'UMAP projection for sentence-embeddings of ASCT+B {asctb_organ}-{asctb_version} Cell-Type annotations'
    )

    fig.show()