from scipy.io import loadmat
import h5py
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import sys
import os

# Input batch size and number of epochs if desired, otherwise defaults to 500 and 100
if len(sys.argv) == 3:
	batchsize = int(sys.argv[1])
	num_epochs = int(sys.argv[2])
else:
	batchsize = 500
	num_epochs = 20

# Load test file
savedir = "train_out"
try:
    os.mkdir(savedir)
except OSError:
    pass

eval_results = savedir + "/ChromDL_train_out.txt"
output = open(eval_results, "w")

batchsize = 500
l1_reg = 1e-8
l2_reg = 5e-7
noutputs = 919
validation_size = 8000
training_size = 4400000

# Load files
f = h5py.File('data/train.mat', 'r')
train_labels_raw = f['traindata'][()]
train_data_raw = f['trainxdata'][()]
train_data = train_data_raw.transpose(2, 0, 1)
trainLabelMatrix = train_labels_raw.transpose(1, 0)
f.close()

y = loadmat('data/test.mat')
testLabelMatrix = y['testdata']
test_data_raw = y['testxdata']
test_data = test_data_raw.transpose(0, 2, 1)

z = loadmat('data/valid.mat')
validLabelMatrix = z['validdata']
valid_data_raw = z['validxdata']
valid_data = valid_data_raw.transpose(0, 2, 1)

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

# Compile and train model for specified number of epochs
filepath = savedir + "/ChromDL_learned_weights"
model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])

# Uncomment this line if loading previously trained weights
# model.load_weights("ChromDL_train_out/ChromDL_learned_weights/variables/variables")
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, verbose=2)
history = model.fit(train_data, trainLabelMatrix, epochs=num_epochs, batch_size=batchsize, shuffle=True, validation_data=(valid_data, validLabelMatrix), callbacks=[checkpointer, earlystopper], verbose=2)

# Save training metrics to file
output.write("Loss\n")
output.write(str(history.history['loss']) + "\n")
output.write("Validation Loss\n")
output.write(str(history.history['val_loss']) + "\n")
output.write("Accuracy\n")
output.write(str(history.history['accuracy']) + "\n")
output.write("Validation Accuracy\n")
output.write(str(history.history['val_accuracy']) + "\n")
output.write("\n")

# Calculate predictions
print("Testing model...", model.evaluate(test_data, testLabelMatrix, batch_size=batchsize, verbose=2))   ### used trained model to the test dataset
predProbs = model.predict(test_data, batch_size=batchsize, verbose=2)   ### get the probability for each testing sequences

# Save predictions to file
f = h5py.File(savedir + "/predProbs.hdf5", "w")
f.create_dataset("pred", data=predProbs)
f.close()

# Calculate auROC/auPRC metrics
aucs = np.zeros(noutputs, dtype=np.float)
auprcs = np.zeros(noutputs, dtype=np.float)
for i in range(noutputs):
        try:
            auc = roc_auc_score(testLabelMatrix[:, i], predProbs[:, i])
            aucs[i] = auc
            aup = average_precision_score(testLabelMatrix[:, i], predProbs[:, i])
            auprcs[i] = aup
        except ValueError:
            pass

histauc, histauprc = aucs[125 + 690:125 + 690 + 104], auprcs[125 + 690:125 + 690 + 104]
dnauc, dnauprc = aucs[:125], auprcs[:125]
tfauc, tfauprcs = aucs[125:125 + 690], auprcs[125:125 + 690]

output.write("auROC list:\n")
output.write(str(list(aucs)))
output.write("\n")
output.write("auPRC list:\n")
output.write(str(list(auprcs)))
output.write("\n")

output.write("\n")
output.write('Transcription factors (' + str(len(tfauc)) + ") samples:\n")
output.write(' - Mean AUC: ' + str(np.nanmean(tfauc)) + "\n")
output.write(' - Mean AUPRC: ' + str(np.nanmean(tfauprcs)) + "\n")

output.write('DNase I-hypersensitive sites (' + str(len(dnauc)) + ") samples:\n")
output.write(' - Mean AUC: ' + str(np.nanmean(dnauc)) + "\n")
output.write(' - Mean AUPRC: ' + str(np.nanmean(dnauprc)) + "\n")

output.write('Histone marks: (' + str(len(histauc)) + ") samples:\n")
output.write(' - Mean AUC: ' + str(np.nanmean(histauc)) + "\n")
output.write(' - Mean AUPRC: ' + str(np.nanmean(histauprc)) + "\n")

output.write('Overall: (' + str(len(tfauc) + len(dnauc) + len(histauc)) + ") samples:\n")
output.write(' - Mean AUC: ' + str(np.nanmean(aucs)) + "\n")
output.write(' - Mean AUPRC: ' + str(np.nanmean(auprcs)) + "\n")
output.close()
