EagleC2
*******
Hi-C has emerged as a powerful tool for detecting structural variations (SVs), but its
sensitivity remains limited—particularly for SVs lacking canonical contact patterns. Here,
we introduce EagleC2, a next-generation deep-learning framework that integrates an ensemble
of convolutional neural networks (CNNs) with diverse architectures, trained on over 2.7
million image patches from 51 cancer Hi-C datasets with matched whole-genome sequencing
(WGS) data. EagleC2 substantially outperforms its predecessor (EagleC) and other state-of-the-art
methods, achieving consistently higher precision and recall across diverse validation datasets.
Notably, it enables the discovery of non-canonical SVs—including complex rearrangements and
fusions involving extremely small fragments—that are frequently missed by existing tools. In
individual cancer genomes, EagleC2 detects over a thousand previously unrecognized SVs, the
majority of which are supported by orthogonal evidence. To support clinical and diagnostic
applications, EagleC2 also offers a rapid evaluation mode for accurately screening predefined
SV lists, even at ultra-low coverage (e.g., 1× depth). When applied to single-cell Hi-C data
from glioblastoma before and after erlotinib treatment, EagleC2 reveals extensive SV heterogeneity
and dynamic structural changes, including events overlooked by conventional pipelines. These
findings establish EagleC2 as a powerful and versatile framework for SV discovery, with broad
applications in genome research, cancer biology, diagnostics, and therapeutic development.

.. image:: ./images/framework.png
        :align: center

Unique features of EagleC2
==========================
Compared with the original EagleC, EagleC2 has the following unique features:

- EagleC2 is able to detect non-canonical SVs, including fine-scale complex rearrangements
  (multiple SVs clustered within a local window) and fusions involving extremely small fragments
- EagleC2 offers a rapid evaluation mode for accurately screening predefined SV lists,
  even at ultra-low coverage (e.g., 1× depth)
- EagleC2 supports arbitrary resolutions, without requiring model re-training for each resolution
- EagleC2 enables fast genome-wide SV prediction on large Hi-C datasets without the need for
  distributed computing across multiple nodes
- EagleC2 supports both CPU and GPU inference

Navigation
==========
- `Installation`_
- `Download pre-trained models`_
- `Overview of the commands`_
- `Quick start`_
- `Visualize local contact patterns around SV breakpoints`_
- `Post-processing and filtering of SV predictions`_
- `Evaluation of predefined SVs`_
- `Clustering of single cells based on EagleC2 predictions`_

Installation
============
EagleC2 and all required dependencies can be installed using `mamba <https://github.com/conda-forge/miniforge>`_
and `pip <https://pypi.org/project/pip/>`_.

After you have installed *mamba* successfully, you can create a conda environment
for EagleC2 by executing the following commands (for Linux users)::

    $ conda config --add channels defaults
    $ conda config --add channels bioconda
    $ conda config --add channels conda-forge
    $ mamba create -n EagleC2 cooler hdbscan numba statsmodels joblib=1.3 numpy=1.26 scikit-learn=1.4 "tensorflow>=2.16"
    $ mamba activate EagleC2
    $ pip install eaglec

This will intall the core dependencies required to run EagleC2.

If you also wish to use the visualization module, please install the following
additional packages::

    $ mamba install matplotlib pyBigWig

If you plan to use the gene fusion annotation module, please install::

    $ mamba install pyensembl

For macOS users (tested on Apple M-series chips only), you can install EagleC2
and its dependencies with::

    $ conda config --add channels defaults
    $ conda config --add channels bioconda
    $ conda config --add channels conda-forge
    $ mamba clean --all
    $ mamba create -n EagleC2gpu python=3.11 hdbscan numba statsmodels joblib=1.3 numpy=1.26 scikit-learn=1.4
    $ mamba activate EagleC2gpu
    $ pip install --no-cache-dir cooler
    $ pip install --no-cache-dir tensorflow==2.16.1 keras==3.3.3
    $ pip install --no-cache-dir tensorflow-metal==1.1.0
    $ pip install eaglec

Similarly, if you would like to use the visualization or gene fusion annotation modules
on macOS, please install *matplotlib*, *pyBigWig*, and *pyensembl* as described above.

Download pre-trained models
===========================
Before proceeding, please download the pre-trained `models <https://www.jianguoyun.com/p/DWhJeUsQh9qdDBjVpoEGIAA>`_ for EagleC2.

Unlike EagleC, which relied on separate models trained for specific resolutions
(e.g., 5 kb, 10 kb, 50 kb, and 500 kb) and sequencing depths, EagleC2 was trained
on a unified dataset that integrates samples across a wide range of resolutions
and depths. This allows for seamless application to data at arbitrary resolutions
and sequencing depths, without the need for model re-training.

Overview of the commands
========================
EagleC2 is distributed with eight command-line tools. You can ``command [-h]`` in a
terminal window to view the basic usage of each command.

- predictSV

  *predictSV* is the core command for predicting SVs from chromatin contact maps.

  Required inputs:

  1. Path to a .mcool file – This is a multi-resolution format for storing contact
     matrices. See `cooler <https://github.com/open2c/cooler>`_ for details. If you only have
     .hic files (see `Juicer <https://github.com/aidenlab/juicer>`_), you can convert them
     to .mcool using `hic2cool <https://github.com/4dn-dcic/hic2cool>`_ or `HiClift <https://github.com/XiaoTaoWang/HiCLift>`_.
  2. Path to the folder containing the pre-trained models.
  
  Output:

  The predicted SVs will be written to a .txt file with 13 columns:

  - Breakpoint coordinates (chrom1, pos1, chrom2, pos2)
  - Probability values for each SV type (++, +-, -+, --, ++/--, and +-/-+)
  - The resolution of the contact matrix from which the SV was originally predicted
  - The finest resolution to which the SV can be refined
  - The number of bad bins near the SV breakpoints

- plot-SVbreaks

  Plots a local contact map centered on the provided SV breakpoint coordinates. For
  intra-chromosomal SVs, contact counts will be distance-normalized. All contact matrices will
  be min-max scaled to the range [0, 1].

  The input breakpoint coordinates should follow the format: "chrom1,pos1,chrom2,pos2".

  This is useful for visually checking whether the expected contact patterns are present
  around SV breakpoints, including those identified by short-read or long-read whole-genome
  sequencing methods.

- filterSV

  Filters the predicted SVs based on the probability values and resolutions with the output .txt files of the *predictedSV*.

- evaluateSV

  Evaluates a predefined list of SVs using EagleC2 models. 

  Required inputs:

  1. Path to a .mcool file.
  2. Path to the folder containing the pre-trained models.
  3. Path to the predifined list of SVs with format 'chrom1 chrom2 breakpoint1 breakpoint2'

  Output:

  The evaluated SVs will be written to a .txt file with 13 columns:

  - Breakpoint coordinates (chrom1, pos1, chrom2, pos2)
  - SV type (++, +-, -+, --, ++/--, or +-/-+)
  - Probability values
  - The resolution of the contact matrix from which the SV was evaluated
  - And if the SV is supported, the supporting coordinates of the Hi-C (pos1, pos2) will be presented.

- reformatSV

  Reformats the output from *predictSV* into a format compatible with `NeoLoopFinder <https://github.com/XiaoTaoWang/NeoLoopFinder>`_.
  
- annotate-gene-fusion

  Annotates gene fusion events for a list of SV breakpoints.

- plot-interSVs

  Plots a contact map for a specified set of chromosomes, with predicted SVs marked.

- plot-intraSVs

  Plots a contact map for a specified genomic region, with predicted SVs marked.

As the commands *annotate-gene-fusion*, *plot-interSVs*, and *plot-intraSVs* are directly
inherited from the original EagleC, this documentation does not cover them in detail. For
more information, please refer to the orignal `EagleC documentation <https://github.com/XiaoTaoWang/EagleC>`_

Quick Start
===========
The following steps will guide you through the process of using EagleC2. All
commands below are expected to be executed in a terminal window.

1. Unzip the pre-trained models
-------------------------------
Place the downloaded pre-trained models in your working directory and unzip the archive::

    $ unzip EagleC2-models.zip

2. Download the test dataset
-----------------------------
Download the test dataset `FY1199.used_for_SVpredict.mcool <https://www.jianguoyun.com/p/DYoL0UgQh9qdDBjdpoEGIAA>`_,
which contains ~18 million contact pairs. This dataset is derived from FY1199,
a human lymphoblastoid cell line with a known balanced inter-chromosomal translocation
between chromosomes 11 and 22 (46,XY,t(11;22)(q23.3;q11.2)). Place the file in the
same directory as the pre-trained models.

3. Run the SV prediction command
--------------------------------
Execute the following command to perform SV prediction on this Hi-C dataset::

    $ predictSV --mcool FY1199.used_for_SVpredict.mcool --resolutions 25000,50000,100000 \
                --high-res 25000 --prob-cutoff-1 0.5 --prob-cutoff-2 0.5 -O FY1199_EagleC2 \
                -g hg38 --balance-type ICE -p 8 --intra-extend-size 1,1,1 --inter-extend-size 1,1,1

For view a full description of each parameter, run::

    $ predictSV -h

What happens when you run the above command
-------------------------------------------
This command performs genome-wide SV prediction on ICE-normalized contact matrices
at 50 kb and 100 kb resolutions (as specified by ``--resolutions``, excluding those
listed in ``--high-res``). To accelerate computation, pixels with significantly
elevated contact counts are identified and extended by 1 bin on both ends (controlled
by ``--intra-extend-size`` and ``--inter-extend-size``; the values specified for these
parameters correspond to each resolution listed in ``--resolutions``) to cover potential
SV breakpoints.

SV predicted at coarser resolutions are progressively refined at higher resolutions.
For example, an SV initially predicted at 100 kb (with a probability cutoff of 0.5,
set by ``--prob-cutoff-1``) will be refined at 50 kb. If the probability at 50 kb exceeds
the second cutoff (set by ``--prob-cutoff-2``), the SV will be further refined at 25 kb.
Otherwise, the 50 kb coordinates are reported as final.

SV predictions across all resolutions are merged in a non-redundant manner. For resolutions
specified in ``--high-res``, the program performs refinement only—not genome-wide scanning.

Computation is parallelized using 8 CPU cores (set via ``-p 8``). If a GPU is available and
the ``--cpu`` flag is not set, model inference will run on the GPU.

Output interpretation
----------------------
After ~5 minutes (depending on your machine), you will find the predicted SVs in a .txt file
named "FY1199_EagleC2.SV_calls.txt" in your working directory::

    $ cat FY1199_EagleC2.SV_calls.txt

    chrom1	pos1	chrom2	pos2	++	+-	-+	--	++/--	+-/-+	original resolution	fine-mapped resolution	gap info
    chr4	52200000	chr4	64400000	0.6095	1.42e-06	1.96e-06	7.996e-09	3.169e-08	9.618e-09	100000	100000	0,0
    chr11	116800000	chr22	20300000	2.495e-11	1.013e-06	5.552e-07	7.897e-11	2.943e-12	1	50000	25000	0,0

The known balanced translocation is successfully detected. The final breakpoint
coordinates (chr11:116800000;chr22:20300000) are reported at 25 kb resolution (see
the "fine-mapped resolution" column), while the SV was initially predicted at 50 kb
(see the "original resolution" column). The last column, "gap info", indicates that
there are no problematic bins in reference genome (hg38, as specified by the parameter
``-g``) near either breakpoint (0,0).

.. note::
    Valid Options for the ``--balance-type`` parameter are "ICE", "CNV" and "Raw".

    - Use "Raw" to process unnormalized contact matrices.
    - Use "ICE" only if your matrices have been balanced with `cooler balance <https://cooler.readthedocs.io/en/latest/cli.html#cooler-balance>`_.
    - Use "CNV" only if your matrices have been CNV-corrected using ``correct-cnv`` from the `NeoLoopFinder <https://github.com/XiaoTaoWang/NeoLoopFinder>`_ toolkit.
    
    Different normalization strategies may yield slightly different results. For best
    sensitivity, we recommend running *predictSV* on all three types of contact matrices
    (Raw, CNV, and ICE) and merging the results.

    All key parameters that may affect the sensitivity and specificity of SV prediction
    are configurable via command-line options. For details, please refer to the help
    message of the *predictSV* command by running ``predictSV -h``.

Prediction command used in the paper
------------------------------------
For the BT-474 Hi-C dataset used in Figure 2, we used the following command
(for HCC1954 and MCF7, the same parameters were used with different `.mcool`
inputs)::

    $ predictSV --mcool BT474.used_for_SVpredict.mcool --resolutions 5000,10000,50000 \
                --high-res 2000 -O BT474_EagleC2 -g hg38 --balance-type CNV \
                -p 8 --intra-extend-size 2,2,1 --inter-extend-size 1,1,1

For the HCC1954 Arima Hi-C dataset used in Figure 3, we used the following command::

    $ predictSV --mcool HCC1954-Arima-allReps-filtered.mcool --resolutions 2000,5000,10000,25000,50000,100000,250000,500000,1000000 \
            --high-res 500 -O HCC1954-Arima -g hg38 --balance-type CNV \
            -p 8 --entropy-cutoff 0.98 --intra-extend-size 3,3,2,2,2,1,1,1,1 \
            --inter-extend-size 2,2,1,1,1,1,1,1,1
and::

  $ predictSV --mcool HCC1954-Arima-allReps-filtered.mcool --resolutions 1000,2000,5000,10000,25000,50000,100000,250000,500000,1000000 \
            --high-res 500 -O HCC1954-Arima -g hg38 --balance-type ICE \
            -p 8 --entropy-cutoff 0.98 --intra-extend-size 3,3,3,2,2,2,1,1,1,1 \
            --inter-extend-size 2,2,2,1,1,1,1,1,1,1

For the single-cell Hi-C datasets used in Figure 6, we used the following command
to predict SVs from pseudo-bulk contact matrices::

    $ predictSV --mcool GBM39-pseudo-bulk.mcool --resolutions 5000,10000,25000,50000,100000,250000,500000,1000000 \
            --high-res 500 -O GBM39-pseudo-bulk -g hg38 --balance-type CNV \
            -p 8 --entropy-cutoff 0.98 --intra-extend-size 3,2,2,2,1,1,1,1 \
            --inter-extend-size 2,1,1,1,1,1,1,1

Visualize local contact patterns around SV breakpoints
======================================================
To assess the quality of predicted SVs, you can visualize the local contact
patterns around the breakpoints using the *plot-SVbreaks* command.

For example, the following command plots the contact map centered on the breakpoints
of the balanced translocation detected in the previous step (see panel a)::

    $ plot-SVbreaks --cool-uri FY1199.used_for_SVpredict.mcool::resolutions/25000 \
                    --balance-type ICE --breakpoint-coords chr11,116800000,chr22,20300000 \
                    --window-width 15 -O chr11,116800000,chr22,20300000.25kb.png --dpi 800

Similarly, this command visualizes the contact patterns around the breakpoints of another  
intra-chromosomal SV detected earlier (see panel b)::

    $ plot-SVbreaks --cool-uri FY1199.used_for_SVpredict.mcool::resolutions/100000 \
                    --balance-type ICE --breakpoint-coords chr4,52200000,chr4,64400000 \
                    --window-width 15 -O chr4,52200000,chr4,64400000.100kb.png --max-value 1 --dpi 800

As shown in the figures, the balanced translocation exhibits a clear butterfly-shaped
contact pattern, consistent with the highest predicted probability for the "+-/-+" SV
type (1). In contrast, the intra-chromosomal SV displays a strong interaction block
in the upper-left quadrant, consistent with the highest predicted probability for the "++"
SV type (0.6095) at the breakpoints.

.. image:: ./images/SVbreaks.png
        :align: center
        :scale: 30%

Post-processing and filtering of SV predictions
===============================================
To refine the SV predictions, you can filter them based on probability values and resolution. The script provides two mutually exclusive filtering strategies.

Filtering Strategies

Resolution-specific filtering: This mode allows you to set distinct probability cutoffs for one or more specific resolutions. You need provide both the --res-cutoffs and --res-list arguments, ensuring that the number of values in each list corresponds to one another.

High/Low resolution filtering: This mode uses a general threshold to filter SVs based on whether they were identified at high resolution (≤10000) or low resolution (>10000). To use this, enable the --high-low-cutoffs flag. You can customize the thresholds with --high-res-cutoff (default: 0.5) and --low-res-cutoff (default: 0.99).

For the filtering of the BT-474 results used in Figure 2, we used the following command (for HCC1954 and MCF7, the same parameters were used with different .txt inputs)::

    $ filterSV -i BT-474_EagleCgpu_2.SV_calls.txt --res-cutoffs 0.5,0.65 --res-list 5000,10000

For the filtering of HCC1954 Arima results used in Figure3-5, we used the following command::

    $ filterSV -i HCC1954_Arima_EagleCgpu_2.SV_calls.txt --high-low-cutoffs --high-res-cutoff 0.5 --low-res-cutoff 0.99

Evaluation of predefined SVs
============================

EagleC2 provides an evaluation mode for rapid and accurate screening of predefined SV lists. Execute the following command to evaluate predefined SV lists with Hi-C dataset::

$ evaluateSV -i HCC1954-SVs.txt -m HCC1954-Arima-allReps-filtered.mcool -o HCC1954 --model-path --resolutions 5000,10000,50000 --balance-type Raw

For view a full description of each parameter, run::

    $ evaluateSV -h

Output interpretation
----------------------

After evaluation, you will find the evaluated SVs in a .txt file named "HCC1954.SV_evaluate.txt" in your working directory::

    $ cat HCC1954.SV_evaluate.txt

    chrom1	pos1	chrom2	pos2	strand	probability	hic_pos1	hic_pos2	resolution
    chr20	12500000	chr9	8000000	--	5.516e-05	12505000	8010000	5000
    chr17	39050000	chr17	72400000	++/--	0.01976	39060000	72395000	5000

Note: SVs with no Hi-C contact signals in the breakpoints' window will not be reported in the .txt files.

Clustering of single cells based on EagleC2 predictions
=======================================================
