from collections import Counter
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt

# todo: Add venn diagram for twoway set-intersection

def make_threeway_venn_diagram(A, B, C, labels, title=''):
    """General function to create a three-way Venn diagram.

    Args:
        A (set): Set of elements
        B (set): Set of elements
        C (set): Set of elements
        labels (list): Names/labels for A, B, C.
        title (str): Defaults to ''.
    """
    fig = plt.figure()
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


    sets = Counter()               #set order A, B, C   
    sets['100'] = len(A_rest)      #100 denotes A on, B off, C off 
    sets['010'] = len(B_rest)      #010 denotes A off, B on, C off
    sets['001'] = len(C_rest)      #001 denotes A off, B off, C on 
    sets['110'] = len(AB_only)     #110 denotes A on, B on, C off
    sets['101'] = len(AC_only)     #101 denotes A on, B off, C on 
    sets['011'] = len(BC_only)     #011 denotes A off, B on, C on 
    sets['111'] = len(ABC_overlap) #011 denotes A on, B on, C on
    plt.figure(figsize=(7,7)) 
    ax = plt.gca() 
    ax.set_title(title, color='white')
    colors = ['darkviolet','deepskyblue','blue']
    v = venn3(subsets=sets, set_labels=labels, ax=ax, set_colors=colors, alpha=0.7)   
    for i, text in enumerate(v.set_labels):
        text.set_color(colors[i])

    for text in v.subset_labels:
        if not text:
            continue
        text.set_color('white')
        text.set_fontsize(16)
        text.set_fontweight('bold')
    plt.savefig(f'data/threeway_intersection_{title.lower()}.png')
    plt.show()
    return fig
    