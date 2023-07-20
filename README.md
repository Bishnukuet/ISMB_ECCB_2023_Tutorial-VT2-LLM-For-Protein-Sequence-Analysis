# ISMB/ECCB 2023 Virtual Tutorial - VT2
# Protein Sequence Analysis using Transformer-based Large Language Model 
To be held as part of  31st Conference on Intelligent Systems For Molecular Biology and 22nd European Conference on Computational Biology, July 23-27, Lyon, France. 
Tutorial Dates: July 17-18, 2023 at 14:00-18:00 hrs CEST  
### Citing this tutorial
Please cite this tutorial as:

**Bishnu Sarker, Sayane Shome, Farzana Rahman, Nima Aghaeepour (2023, July). Tutorial VT2: Protein Sequence Analysis using Transformer-based Large Language Model. In 31st Conference on Intelligent Systems for Molecular Biology and 22nd European Conference on Computational Biology (ISMB/ECCB 2023), Lyon, France.**

### Overview
In the current decade, Artificial Intelligence (AI) / Machine Learning(ML) has tremendously facilitated scientific discoveries in biomedicine. Moreover, the recent advancements in the development of large language models (a type of deep learning model that can read, summarize, translate, and generate text as we humans do) have inspired many researchers to find applications in biological sequence analysis, partly because of the similarities in the data. Attention-based deep transformer models [1,2] pre-trained in a self-supervised fashion on large corpus have dramatically transformed research in natural language processing. The attention mechanism involved in transformer models captures the long-distance relationship among words in textual data [2]. Following a similar principle in the biological domain, researchers have trained transformer-based protein language models for biological sequence analysis. For example, ProtTrans [3] was trained on UniProtKB [4] sequences for protein sequence analysis. They showed that transformer-based self-supervised protein language models effectively capture the spatial relationship among residues which is critical for understanding the functional and structural aspects of proteins. 

In this 8-hour long (divided into two sessions: July 17-18, 2023 from 14:00 to 18:00 hrs CEST) online tutorial, we aim to provide experiential training on how to build basic ML pipelines using deep learning and pre-trained transformer protein language models for biological sequence analysis. We will start with a quick introduction to Python packages (Keras, Tensorflow/Pytorch, Scipy, scikit-bio, bio-transformers) that are heavily used for ML projects. In addition, we will cover the biological concepts behind protein sequence and function. Then, we will introduce classical natural language processing, and report its recent advancements. Finally, self-supervised deep learning-based large language models (such as Transformers) will be reviewed with a particular focus on protein sequence analysis. 
References 
1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
3. Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Yu, W., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M. and Bhowmik, D., 2021. ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High-Performance Computing. IEEE Transactions on Pattern Analysis and Machine Intelligence. 
4. UniProt Consortium, 2007. The universal protein resource (UniProt). Nucleic acids research, 36(suppl_1), pp. D190-D195. 

### Learning Objectives 
At the end of the tutorial, the participants will have understanding and practical knowledge of: 

1. How to collect, preprocess and vectorize sequence data. 
2. How to build basic ML models for sequence analysis.
3. How to implement deep learning models such as Long-Short Term Memory and Recurrent Neural Networks (LSTM and RNN) in the context of biological sequence modelling.
4. Fundamentals of transformer-based large language models. 
5. How to apply a pre-trained transformer language model for biological sequence analysis 
6. How to formulate and address biomedical problems using transformer-based large language models. 
7. What tools, frameworks, datasets, and programming libraries are available to work with transformer-based large language models for sequence analysis.

### Target Audience
The target audiences are graduate students, researchers, scientists, and practitioners in both academia and industry who are interested in applications of deep learning, natural language processing, and transformer-based language models in biomedicine and biomedical knowledge discovery. The tutorial is aimed towards entry-level participants with basic knowledge of computer programming (preferably Python) and machine learning (Beginner or Intermediate). 

### Instructions
The participants are requested to follow the following steps to prepare their work environment. 

#### Minimum Requirements:

- A computer
- High speed internet. 
- A Google Drive account. Make sure you can access Google Drive and Google Colaboratory. 
- Minimum working knowledge of Python, particularly basic looping, lists, array, tensors. A refresher on Python will be provided as a part of the tutorial. However, we recommend a quick refresher  on PyTorch and Scikit-Learn  Python packages for ML model development. 

#### Setting up the environment:

- Create a folder named ISMB_ECCB_2023  in your Google Drive.
- Upload the [data](https://drive.google.com/drive/folders/1118He3vsn-mwMoRDsmWPG_iEzrZ68Vda?usp=sharing)  Folder shared with you and place it under the folder ISMB_ECCB_2023. With data folder placed correctly, the notebooks should be executable without any error. 
- Upload ISMB_ECCB_2023_notebooks  folder shared with you and place it under ISMB_ECCB_2023 .

# Outline

## DAY 1 - July 17, 2023 (14:00 -18:00 hrs CEST)

### 14:00-14:30 hrs CEST | Introduction | [Slides](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/Slides/Lecture%201.1%20-%20Introduction%20and%20Fundamentals%20of%20Protein%20Biology.pdf)
- Introduction to the tutorial session
- Fundamental concepts about proteins from a biological perspective  

### 14:30-14:45 hrs CEST  - 15 minutes Break/Q&A
### 14:45-15:45 hrs CEST | Python Programming Refresher | [Slides](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/Slides/Lecture%201.2%20-%20Programming%20Basics%20with%20Python.pdf)
- Basic Syntax  
- Main Python packages to perform scientific computation such as NumPy/Scipy, Pandas, Matplotlib, Biopython. 
- ML packages such as Scikit-Learn, PyTorch.

[Colab-Notebook-Python-Refresher](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/notebooks/Day%201-Part-1A-Python-Refresher.ipynb)

### 15:45-16:00 hrs CEST  - 15 minutes Break/Q&A

### 16:00-17:45 hrs CEST | Introduction to biological sequence analysis using Deep Learning in Python | [Slides](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/Slides/Lecture%201.3%20-%20Introduction%20to%20Sequence%20Modeling.pptx.pdf)
- Introduction to biological sequence analysis using Deep Learning in Python
- Building deep learning models (RNN, LSTM) for sequence analysis

[Colab-Notebook-RNN](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/notebooks/Day%201-Part-1B-RNN_Sequence_Classification.ipynb)  [Colab-Notebook-LSTM](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/notebooks/Day%201-Part-1C-LSTM_Sequence_Classification.ipynb)

### 17:45-18:00 hrs CEST  - 15 minutes Break/Q&A


## DAY 2 - July 18, 2023 (14:00 -18:00 hrs CEST)

### 14:00-15:00 hrs CEST | Introduction to Transformer-Based Language Model | [Slides](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/Slides/Lecture%202.1%20-%20Introduction%20to%20Transformer-Based%20Language%20Model.pptx.pdf)
-Introduction to Transformer-based Language Models 
-Transformers for biological sequence analysis 

[Colab-Notebook-Transformer](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/notebooks/Day%202-Part-2A-Loading-ProtT5-Pre-Trained-Embeddings.ipynb)

### 15:00-15:15 hrs CEST  - 15 minutes Break/Q&A

### 15:15-16:30 hrs CEST | Hands-On Case Study 1 - Protein Function Annotation | [Slides](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/Slides/Lecture%202.2%20-%20Case%20Studies_and_Closing_Remarks.pdf)
-Introduction to the Case Study-1: Protein function Annotation
-Building a protein function annotation pipeline using Transformer-based language model.  

[Colab-Notebook-Case-Study-1-Protein-Function-Prediction](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/notebooks/Day%202-Part-2B-Case_Study1-Transformer-Protein-Sequence-Classification.ipynb)

### 16:30-16:45 hrs CEST  - 15 minutes Break/Q&A

### 16:45-17:45 hrs CEST | Hands-On Case Study 2 - Protein Metal-Binding Site Prediction | [Slides](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/Slides/Lecture%202.2%20-%20Case%20Studies_and_Closing_Remarks.pdf)
-Introduction to Case Study-2:  Protein Metal-Binding Site Prediction.  
-Building a protein-metal binding site prediction pipeline using Transformer-based language model.  

[Colab-Notebook-Case-Study-2-Metal-Binding-Site-Prediction](https://github.com/Bishnukuet/ISMB_ECCB_2023_VT2_LLM/blob/main/notebooks/Day%202-Part-2C-Case_Study2_Transfomer_Metal_Binding_Site_Prediction.ipynb)

### 17:45-18:00 hrs CEST  - 15 minutes Q&A and closing remarks


### Organizers: 
#### 1. Bishnu Sarker 
Bishnu Sarker is an Assistant Professor of Computer Science and Data Science at Meharry Medical College, Nashville, TN, USA. His research focus is on applying AI, deep learning, natural language processing (NLP), and graph-based reasoning approaches to effectively describe proteins numerically and to infer their functional characteristics from complex, heterogeneous, and interconnected biomedical data. He received his BS from Khulna University of Engineering and Technology, Bangladesh; MS from Sorbonne University, France; and PhD from INRIA, France. During his PhD he spent a winter at MILA - Quebec AI Institute and University of Montreal, Canada, as a visiting researcher with the DrEAM mobility grant from University of Lorraine. 

#### 2. Sayane Shome 
Sayane Shome is a postdoctoral researcher at Stanford University, USA, as a joint appointment between Dr. Nima Aghaeepour and Dr. Lawrence S. Prince's labs. She completed her PhD. in Bioinformatics from Iowa State University, USA, under the supervision of Dr. Robert L. Jernigan. Her Ph.D. dissertation involved studying various membrane protein systems using computational biophysical methods. Under her postdoctoral fellowship at Stanford, she has been working with computational analysis of different omics datasets and developing AI models using biomedical data focussed on understanding the progression of Bronchopulmonary dysplasia in newborn babies. In addition, she has strong interests in outreach, science communication, and STEM awareness amongst underrepresented groups and has volunteered with various non-profit organizations. She has also been associated with the International Society of Computational Biology (ISCB)-Student Council at leadership levels for several years. 

#### 3. Farzana Rahman 
Farzana Rahman is a Lecturer ( USA Level: Assistant Professor) of Computer Science at Kingston University London, UK. Her research focuses on evolutionary genomics, proteomics and natural crisis modelling using ML, deep learning, and cloud computing. She is actively involved in improving computational pedagogy utilising Wikipedia knowledge base. She is an open-source science advocate and an experienced international STEM conference organizer. She is a co-chair of the International Society for Computational Biology's (ISCB) Wikipedia Committee and Editing Competition. She is also a founding member of the ISCB publication committee. As part of her leadership role, Farzana served a 3-year term as an elected Board of Director at the ISCB. 

#### 4. Nima Aghaeepour 
Nima Aghaeepour is an Associate Professor at Stanford University. His laboratory develops ML and AI methods to study clinical and biological modalities in translational settings. He is primarily interested in leveraging multi-omics studies, wearable devices, and electronic health records to address global health challenges. His work is recognized by awards from numerous national and international organizations, including the Bill and Melinda Gates Foundation, the Alfred E. Mann Foundation, the March of Dimes Foundation, the Burroughs Wellcome Fund, and the National Institute of General Medical Sciences. 


### Acknowledgements
We would like to thank all the participants for attending this tutorial session and making it a successful event. 
Greatly indebted to ISMB/ECCB 2023 Tutorial committee for accepting this tutorial for presentation. Also very thankful to the technical team of ISCB for handling issues quickly during the tutorial session.
Would like to thank School of Applied Computational Sciences at Mehary Medical College, TN, USA for the logistics support to present this tutorial. 
Would like Dr. Sayane Shome and Dr. Nima Aghaeepour  from Standford School of Medicine, and , Dr. Farzana Rahman from Kingston University, UK for the tremendous effort in preparation of these materials.  
