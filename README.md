# WYSIWIM: Semantic Representation Learning of Code based on Visualization and Transfer Learning


## Description:  
WYSIWIM ("What You See Is What It Means") is a novel approach to learning semantic representations of code via transfer learning, based on visual representations of source code.  

![Transfer Learning](/images/transfer_learning.png)  

We asses our approach on two different semantic learning tasks, namely **code classification** and **semantic code clone detection**.  

The *code classification* consists of attributing a functionality label to a code fragment, out of a pre-defined finite set of possible labels.  
As depicted directly below, we implement this approach by visualizing our code as an image and use the resulting images to retrain a pre-trained image classification neural network, namely ResNet (ResNet18 and ResNet50 for comparison). You may refer to the original paper for further implementation details or have a direct look at the code.  


![Code Classification Architecture](/images/architecture_cc.png)  


The *semantic code clone detection task* consists of deciding, given two code fragments, whether or not they represent a pair of semantic clones.  
As depicted directly below, we implement this approach by visualizing both code fragments as an image respectively, then we use a pre-trained ResNet50 image classification neural network again but remove the last layer in order to extract the internal feature vectors of the network. Those feature vectors are then used to train one of three different classical binary classification algorithms (Support Vector Machine, k Nearest Neighbours and a simple neural network). Again for further implementation details, please refer to our original paper.  

![Code Clone Detection Architecture](/images/architecture_ccd.png)  


## Contents of this repository  
This repository includes the artifacts used for our experiments, i.e. the implementation for both tasks presented above.  
The structure of the repository is as follows:  
.  
. license.txt                      : The GPLv3 license for our implementation EXCEPT   for the datasets (see below)  
├── astnn_changes                  : Changes on the pipeline we did to ASTNN for our experiments (i.e. to allow the automated extraction of results)  
├── clone_detection                : Implementation of the clone detection task  
│   ├── datasets                   : Our custom select datasets, based on the BCB  
│   │   ├── ds_no_duplicates \*     : The version of our dataset without clone duplicates  
│   │   └── ds_with_duplicates \*   : The version of our dataset we used for most experiments on CCD  
│   └── learning  
│       └── astnn                  : Contains a scripts to automatically convert our dataset into the format of ASTNN and runs the later on it  
├── code_classification            : Implementation of the code classification task  
│   └── datasets                     
│       └── oj \*\*                  : Contains a compressed version of the OpenJudge dataset converted to our input format   
├── images                         : Contains images used in this readme  
└── visualization_algorithms       : Contains the code for generating our visual representations  
    ├── ast_simple                 : The condensed AST rendering algorithm  
    ├── keywords_picto             : The geometric syntax highlighting rendering algorithm  
    │   └── keyword_forms          : The icons that are applied for the respective keywords  
    │       └── unused             : Those are ignored by the implementation  
    ├── simpletext                 : The simple b/w textual rendering algorithm  
    └── synth_high                 : The classical color syntax highlighting rendering algorithm  
    
    
**Licenses:**  
\* The datasets based on BCB are provided under the GPLv2 license as provided by the original authors Svajlenko et al.: (https://github.com/jeffsvajlenko/BigCloneEval)  

\*\* The Open Judge dataset is provided under the license as meant by the original authors, Mou at al.: (https://sites.google.com/site/treebasedcnn/)  
> All the material can be used freely for non-commercial purposes. [...]

The rest of our implementation is provided under GPLv3 license (see license.txt)  


## Running the code  
### Prerequisites  

NOTE: all of the below was executed on an Ubuntu 18.04 installation, compatibility with other OSes cannot be guaranteed  
  
* Pre-requisites: an installed ASTNN repository: https://github.com/zhangj111/astnn  
  
<astnn_path> represents the installation path of your ASTNN repository  
<bceval_path> represents the installation path of your BigCloneEval repository  
<repo_path> representst the installation path of the current repository  
  
### Use included datasets:  
  
To run Code Classification:  
 - Unpack ./code_classification/datasets/oj/fragments.tar.gz to fragments.csv in the same folder  
 - Replace <repo_path> in DATASETS_PATH in file ./code_classification/experiments.py by repository main folder path  
 - Replace <repo_path> in ds_path in file ./code_classification/visualize_all.py   
 - Run the visualizations (from the code_classification folder as working dir):   
    python3 visualize_all.py  
 - Run the experiment (from the code_classification folder as working dir):   
    python3 experiments.py  
  
To run Code Clone Detection:  
 - Download ASTNN sourcecode  
 - Backup original and replace file <astnn_path>/clone/pipeline.py by astnn_changes/pipeline.py  
 - Backup original and replace file <astnn_path>/clone/train.py by astnn_changes/train.py  
 - Replace <repo_path> in datasets_path in file ./clone_detection/experiments.py by repository main folder path  
 - Replace <repo_path> in ds_path in file ./clone_detection/visualize_all.py  
 - Replace <repo_path> in KEYWORD_PICS_PATH in file ./visualization_algorithms/keywords_picto/alg.py  
 - Replace <astnn_path> in astnn_path in file ./clone_detection/learn.py  
 IMPORTANT: make sure to update this path correctly, as it is used to remove temporary/cache files using the rm command!  
  
 - Run the visualizations (from the clone_detection folder as working dir):   
    python3 visualize_all.py  
 - Run the experiment (from the clone_detection folder as working dir):   
    python3 experiments.py  
  
  
  
### Recreate datasets from scratch: (the datasets are included but those steps allow to recreate them)  
Code Classification:  
 - Retrieve the original OJ dataset by Mou et al. from: https://sites.google.com/site/treebasedcnn/  
 - Use the following jupyter notebook to transform the dataset to our format: (you may need to change the paths; the fragments.csv should be in the  ./code_classification/datasets/oj folder later)  
   ./code_classification/datasets/generate_OJ_dataset.ipynb  
  
Clone Detection:  
 - Install and run ASTNN first (to generate their dataset), as described in: https://github.com/zhangj111/astnn  
 - Download and install the BCB as described in BCBEval repository (https://github.com/jeffsvajlenko/BigCloneEval)  
 - Retrieve the H2 database utility (the .jar version) from: https://www.h2database.com/html/main.html  
 - Run the BCB database using the following command: (replace the <bceval_path> in the command; make sure that the H2 utility is in the working directory or add it explicity to the java path upon execution)  
    java -cp h2-*.jar org.h2.tools.Server -baseDir <bceval_path>/bigclonebenchdb -ifExists  
 - Connect to the database via the Webbrowser and change the password of the user "sa" to "sa" using the following query: (you might want to clear it later if you plan to use the BCB)  
    ALTER USER sa SET PASSWORD 'sa';  
 - Replace <repo_path> in file ./clone_detection/datasets/extract_astnn_dataset.py  
 - Replace <repo_path> and <bcb_path> in file ./clone_detection/datasets/extract_vc_dataset.py  
 - Run the DB creation scripts (from the clone_detection/datasets folder as working dir):  
    python3 extract_astnn_dataset.py  
    python3 extract_vc_dataset.py  
    
