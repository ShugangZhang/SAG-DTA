# SAG-DTA
The code is the implementation for the paper 'SAG-DTA: Prediction of Drug–Target Affinity Using Self-Attention Graph Network'.

### Requirements
python            3.6.7<br/>
pytorch           1.2.0<br/>
scipy             1.3.1<br/>
numpy             1.17.2<br/>
pandas            0.25.1<br/>
deepchem          2.2.1<br/>
pickle            0.7.5<br/>
rdkit             2019.03.4.0<br/>
sklearn           0.0.0<br/>

### Data Download
Please download the data file (SAG_DTA_data.zip) from the following BaiduDisk link.

Download link: https://pan.baidu.com/s/1AHy6gcqW9H1lt6CrnN7DLw<br/>
Extraction code: zy2n

Uncompress the file to get a 'data' folder containing all the original data and processed data. Replace the original 'data' folder by this new folder.

### Run the code
Run the 'predict_with_pretrained_model_BindingDB/Human' for a quick check of the results reported in the paper. Or if you want to train the network by yourself, run the 'training_validation_BindingDB/Human/Davis_KIBA'. The training process should be less than 1 hour.

### Create a novel network
The network files for SAG-Global and SAG-Hierarchical structures are within the 'models' folder. If you wish to create a novel network by yourself, you can do so by adding new network file in this folder. Noted that there are also some other networks from the GraphDTA (see https://github.com/thinng/GraphDTA) for you references.

### Create a new dataset
You might like to test the model on more DTA or CPI datasets. If this is the case, please add the data in the folder 'data' and process them to be suitable for PyTorch. Detailed processing scripts for converting original data formats to PyTorch formats have been uploaded for your references, e.g., prepare_data_Human.py, prepare_data_bindingDB.py.

### Acknowledging this work
If you publish any work based on the contents of this repository please cite:
Zhang, S.; Jiang, M.; Wang, S.; Wang, X.; Wei, Z.; Li, Z. SAG-DTA: Prediction of Drug–Target Affinity Using Self-Attention Graph Network. Int. J. Mol. Sci. 2021, 22, 8993. https://doi.org/10.3390/ijms22168993
