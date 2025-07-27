# Methods Summary: A spatially resolved atlas of the human lung...

This summary focuses on the *Methods*, *Reporting Summary* and parts of the *Key Resources Table* sections of the paper.

---

**Overall Experimental Design and Approach**
The study combines multiple, complementary single-cell and spatial transcriptomics methods to create a comprehensive atlas of the human lung. The key design elements were:

*   **Multi-location sampling:** Tissue samples were taken from five distinct locations along the respiratory tract: trachea, bronchi (2nd-3rd generation), bronchi (4th generation), upper parenchyma, and lower parenchyma.  This allows for comparison of cell types and states across different anatomical regions.
*   **Multi-omic approach:**  The study *integrates* data from:
    *   **scRNA-seq:** Single-cell RNA sequencing (of cell suspensions).
    *   **snRNA-seq:** Single-nucleus RNA sequencing (of nuclei suspensions, allowing analysis of cell types difficult to isolate intact, like chondrocytes).
    *   **VDJ-seq:** Sequencing of T cell receptor (TCR) and B cell receptor (BCR) genes to assess clonality.
    *   **Visium Spatial Transcriptomics (ST):**  Spatial gene expression profiling to map cell types and states within tissue sections.
    * **smFISH**
    * **IHC**: immunohistochemistry
*   **Integration with existing data:** The study integrates its findings with existing lung cell atlases (HLCA, LungMAP) and other published datasets.
*   **Computational analysis:** Sophisticated computational methods were used for data integration, cell type annotation, differential gene expression analysis, trajectory analysis, and spatial mapping.

---

**METHODS DETAILS (From Methods, Reporting Summary and Key Resources)**

**1. Tissue Acquisition and Processing:**

*   **Source:** Deceased transplant organ donors (with informed consent).
*   **Locations:** Trachea, bronchi (2nd-3rd generation), bronchi (4th generation), upper parenchyma, lower parenchyma.
*   **Processing:**
    *   Fresh tissue: Dissociation into single-cell suspensions.  Enzymatic digestion protocols varied (Liberase/trypsin/collagenase, or collagenase D). Some samples had CD45+ cell enrichment.
    *   Frozen tissue: Nuclei isolation for snRNA-seq.  Tissue embedded in OCT and frozen for spatial transcriptomics.
    *   FFPE (formalin-fixed paraffin-embedded) tissue:  Used for Visium ST and validation experiments.

**2. Single-Cell/Nuclei RNA Sequencing (scRNA-seq/snRNA-seq):**

*   **Platforms:** 10x Genomics Chromium (multiple chemistries).
*   **Sequencing:** Illumina platforms (NextSeq 500, NovaSeq 6000).
*   **Data Processing:**
    *   Cell Ranger (10x Genomics) for initial processing.
    *   SoupX for ambient RNA correction.
    *   Scanpy for quality control, normalization, integration, and clustering.
    *   Harmony, BBKNN, or scVI-tools for batch correction/data integration.
    *   Doublet detection and removal (using multiple methods).
    * Manual annotation of cell types with existing data.

**3. V(D)J Sequencing:**

*   TCR and BCR sequencing from single-cell libraries.
*   Scirpy used for analysis.

**4. Spatial Transcriptomics (Visium ST):**

*   **Platform:** 10x Genomics Visium.
*   **Tissue Preparation:** Fresh frozen or FFPE tissue sections.
*   **Data Processing:** Space Ranger (10x Genomics), cell2location for mapping cell types from scRNA-seq data onto spatial spots.

**5. smFISH (Single-Molecule Fluorescence In Situ Hybridization):**

*   **Platform:** RNAscope.
*   **Probes:** Designed to target specific mRNAs.

**6. Immunohistochemistry (IHC):**

*  Multiple rounds of staining
* Standard protocols.

**7. Imaging:**

*   **Confocal microscopy:** Zeiss LSM 880.
*   **Two-photon microscopy:** LaVision.
* **Multiphoton Microscopy**
* **Slide Scanner**

**8. Image Analysis:**

*   **Imaris (Bitplane):**  For cell tracking, spot detection, and quantitative analysis.
*   **FIJI/ImageJ:**  For image processing.

**9. Computational Analyses:**

*   **Differential Gene Expression:**  Wilcoxon rank-sum test, Mann-Whitney U test, Poisson linear mixed model.
*   **Gene Set Enrichment Analysis (GSEA):**  g:GOSt method and g:Profiler.
*   **Cell-Cell Interaction Analysis:**  CellChat.
*   **Trajectory Analysis:**  scVelo, Monocle3, Palantir.
*   **Spatial Mapping:**  cell2location.
*   **Data Integration:** Harmony, BBKNN, scVI-tools.
*  **Pseudotime analysis**
* **Non-negative Matrix Factorization**

**10. Statistical Analyses:**

*   Spearman's rank correlation test.
*   Wilcoxon rank-sum test.
*   Mann-Whitney U test.
*   One-way ANOVA with Tukey post-hoc tests.
*   Poisson linear mixed model.
*   GraphPad Prism and R were used for statistical computations.

---
**KEY RESOURCES TABLE (Partial - Due to Length)**

The Key Resources Table is *extensive* and lists:

*   **Antibodies:**  A very long list of antibodies used for flow cytometry, immunohistochemistry, and other assays.  Includes clone names, vendors, and dilutions.
*   **Bacterial and Virus Strains:**
*   **Chemicals, Peptides, and Recombinant Proteins:** A detailed list of reagents used in the study.
* **Critical Commercial Assays:**
*   **Deposited Data:**  Links to publicly available datasets.
*   **Experimental Models: Organisms/Strains:**
    *   Human tissue samples.
* **Software and Algorithms:**
     *  Cell Ranger.
     * Scanpy.
     * SoupX.
     * Harmony.
     * BBKNN
     * scVI-tools
     * CellTypist
     * Imaris
     * FIJI/Imagej
     * R Studio
    *  Space Ranger
    * Cell2location
    * Omero

---
**Reporting Summary**
The authors detail their adherence to reporting guidelines, including study design, sample size, inclusion/exclusion criteria, etc., as per journal requirements.

The summary above organizes and extracts the core methodological information, providing a clear overview of the techniques and resources employed in this comprehensive study. Due to the length constraints, I have only provided a partial overview of the "Key Resources Table." The full table in the original paper is essential for complete reproducibility.