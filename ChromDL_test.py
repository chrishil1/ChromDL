from scipy.io import loadmat
import h5py
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import sys
import os

# Choose weight file
if len(sys.argv) == 2:
	weight_file = str(sys.argv[1])
else:
	weight_file = "ChromDL_best_weights"

# Create output directory
savedir = "test_out"
try:
    os.mkdir(savedir)
except OSError:
    pass

output = open(savedir + "/ChromDL_test_metrics.txt", "w")

# Initialize parameters
batchsize = 500
l1_reg = 1e-8
l2_reg = 5e-7
noutputs = 919

# Load test file
y = loadmat('data/test.mat')
testLabelMatrix = y['testdata']
test_data_raw = y['testxdata']
test_data = test_data_raw.transpose(0, 2, 1)

# Define model
model = tf.keras.models.Sequential([
	tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True), input_shape=(1000,4)),
	tf.keras.layers.SeparableConv1D(750, (16), activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
	tf.keras.layers.Convolution1D(360, (8), activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
	tf.keras.layers.MaxPooling1D(4),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.AveragePooling1D(8),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(noutputs, activation='sigmoid')
])

# Load best model weights and calculate predictions
model.load_weights(weight_file + "/variables/variables")
model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])
predProbs = model.predict(test_data, batch_size=batchsize, verbose=2)

# Save predictions to file
f = h5py.File(savedir + "/predProbs.hdf5", "w")
f.create_dataset("pred", data=predProbs)
f.close()

# Calculate auROC/auPRC metrics
aucs = np.zeros(noutputs, dtype=float)
auprcs = np.zeros(noutputs, dtype=float)
for i in range(noutputs):
        try:
            aucs[i] = roc_auc_score(testLabelMatrix[:, i], predProbs[:, i])
            auprcs[i] = average_precision_score(testLabelMatrix[:, i], predProbs[:, i])
        except ValueError:
            pass

histauc, histauprc = aucs[815:], auprcs[815:]
dnauc, dnauprc = aucs[:125], auprcs[:125]
tfauc, tfauprc = aucs[125:815], auprcs[125:815]

# Save testing results
output.write(f"auROC list:\n{str(list(aucs))}\n")
output.write(f"auPRC list:\n{str(list(auprcs))}\n\n")

output.write(f"Transcription factors ({str(len(tfauc))}) samples:\n")
output.write(f" - Mean AUC: {str(np.nanmean(tfauc))}\n")
output.write(f" - Mean AUPRC: {str(np.nanmean(tfauprc))}\n")

output.write(f"DNase I-hypersensitive sites ({str(len(dnauc))}) samples:\n")
output.write(f" - Mean AUC: {str(np.nanmean(dnauc))}\n")
output.write(f" - Mean AUPRC: {str(np.nanmean(dnauprc))}\n")

output.write(f"Histone marks: ({str(len(histauc))}) samples:\n")
output.write(f" - Mean AUC: {str(np.nanmean(histauc))}\n")
output.write(f" - Mean AUPRC: {str(np.nanmean(histauprc))}\n")

output.write(f"Overall: ({str(len(tfauc) + len(dnauc) + len(histauc))}) samples:\n")
output.write(f" - Mean AUC: {str(np.nanmean(aucs))}\n")
output.write(f" - Mean AUPRC: {str(np.nanmean(auprcs))}\n")

output.close()
