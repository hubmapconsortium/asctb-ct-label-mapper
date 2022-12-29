from setuptools import setup

setup(
    name='asctb_ct_label_mapper',
    version='0.0.1',    
    description="A package to ensure controlled vocabulary for annotations of scRNA-seq datasets. The goal is to enable cross-dataset or cross-experiment comparison of data by aligning annotations to a standard reference point.",
    url='https://github.com/hubmapconsortium/asctb-ct-label-mapper',
    author='Vikrant Deshpande',
    author_email='vikrant.deshpande09876@gmail.com',
    packages=['asctb_ct_label_mapper'],
    install_requires= [
        "setuptools>=61.0", 
        "scikit-learn==1.0.2", 
        "pandas==1.4.2", 
        "nltk==3.7",
        # "torch==1.13.1",
        # "torch@https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp39-cp39-win_amd64.whl"
        "contractions",
        # "sentence-transformers==2.2.2",
        "matplotlib-venn==0.11.7"
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.7',
    ]
)