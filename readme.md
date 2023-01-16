# ChromDL: A Next-Generation Regulatory DNA Classifier
ChromDL is a hybrid neural network comprised of bidirectional gated recurrent Units (BiGRUs), separable and traditional convolutional neural networks (CNNs), and bidirectional long short-term memory units (BiLSTMs) for the prediction of Transcription Factor Binding Sites (TFBS), DNase-I hypersensitive sites (DHS), and Histone Modifications (HM) using only DNA sequences as input.

![model image](ChromDL_vis.png)

## Requirements
ChromDL was trained and run on the Biowulf HPC Cluster hosted by the National Institutes of Health. To do so, a .sif container file was created that included all required packages and can be found in the container definition file. The docker image was compiled from the Nvidia CUDA 11.0 toolkit including cuDNN version 8 for Ubuntu 18.04. 

Included in this container are the following:
- Python (version 3.7.5)
- Tensorflow-GPU (version 2.4.0 from Google API Linux releases)
- H5py (version 2.10.0)
- Numpy (version 1.19.5)
- Scikit-learn (version 0.24.1)
- SciPy (version 1.6.0)
- Sklearn (version 0.0)

Also included in the container file for other file processing were:
- Pandas (version 1.2.1)
- Tensorflow Addons (version 0.12.1)
- Matplotlib (version 3.3.4)

ChromDL was trained for 100 epochs but stagnanted after 20 epochs, with each epoch taking ~16 hours on a single core NVIDIA Tesla K80 GPU processor with 24 GB RAM.

ChromDL can also be run out of the box locally or on a Google Colaboratory GPU, where `ChromDL_test.py` can be run using a standard GPU runtime, and `ChromDL_train.py` using a high-RAM premium GPU runtime.

## Usage
### Training ChromDL
`python ChromDL_train.py`

ChromDL_train.py will train the model for the 100 epochs using a batch size of 500 and save results to `train_out/ChromDL_train_out.txt`, and the weights from the training to `train_out/ChromDL_learned_weights`

Or, the user can input the number of epochs or change the batch size and run:

`python ChromDL_train.py <batch size> <number of epochs>`

ChromDL_train.py will train the model for the specified number of epochs using the inputted batch size and save results to `train_out/ChromDL_train_out.txt`, and the weights from the training to `train_out/ChromDL_learned_weights`

### Testing ChromDL
`python ChromDL_test.py`

ChromDL_test.py will run the fully trained model loaded from `ChromDL_best_weights` and calculate the performance metrics for auROC and auPRC, saving them to `test_out/ChromDL_test_metrics.txt`. Alternatively, if you wish to calculate training metrics on a user trained model, run the following:

`python ChromDL_test.py <weight file>`

Where the weight file will be the `ChromDL_learned_weights` from training.

There is also a ChromDL_test.ipynb interactive notebook that can be run on google Colaboratory using the free GPU runtime and the testing dataset.

## Associated Files
### Data
The ChIP-seq and DNase-seq data that this model was trained on can be found in DeepSEA's codebase, <http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz>.

The individual links for each of the 919 chromatin features labels can be found in `data/data_sources/DeepSEA_data_sources.txt`, and the other data sources used in the enhancer and raQTL experimentation and validation of the model can be found in `data/data_sources/`

### Container file
The container.sif used as the docker image for training and testing ChromDL can be downloaded from: https://drive.google.com/uc?id=1LHVtgATsO4n4WYCFdw46goFgjibCrjip&export=download

### Weight files
The fully trained ChromDL weight files can be downloaded from:
https://drive.google.com/drive/folders/1vWxdDRbn3v4oit5D9RiMxiR-vopYnj6w?usp=sharing

## Contact information
For help or issues using ChromDL, please contact chrishil@umich.edu.
