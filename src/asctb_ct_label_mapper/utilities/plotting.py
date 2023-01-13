from collections import Counter
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt


def make_venn_diagram(A, B, C=None, labels=['A','B','C'], title=''):
    """General function to create a two-way or three-way Venn diagram.

    Args:
        A (set): Set of elements
        B (set): Set of elements
        C (set, optional): Set of elements. Defaults to None.
        labels (list, optional): Names/labels for A, B, C. Defaults to ['A','B','C'].
        title (str): Defaults to ''.
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
    plt.savefig(f'data/threeway_intersection{"_"+title if title else ""}.png')
    plt.show()