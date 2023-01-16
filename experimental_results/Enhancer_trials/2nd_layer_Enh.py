import h5py
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from tensorflow.keras.constraints import max_norm
import sys

# Files named "E003-ChromS_Enh.bed" depending on enhancer definition
filename = sys.argv[1]

batchsize = 500
l1_reg = 1e-8
l2_reg = 5e-7
noutputs = 919
epoch_num = 100

# File load
f = h5py.File(filename, 'r')
train_data_in = f['train_data'][:]
train_data = train_data_in[:int(len(train_data_in) / 2)]
test_data = f['test_data'][:]
valid_data = f['validation_data'][:]
trainLabelMatrix_in = f['train_labels'][:]
trainLabelMatrix = trainLabelMatrix_in[:int(len(trainLabelMatrix_in) / 2)]
testLabelMatrix = f['test_labels'][:]
validLabelMatrix = f['validation_labels'][:]

# ChromDL model definition
chromDL = tf.keras.models.Sequential([
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

chromDL.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])
chromDL.load_weights("ChromDL_best_weights/variables/variables")

# Feed datasets through ChromDL first
trainProbs = chromDL.predict(train_data, batch_size=batchsize, verbose=1)
testProbs = chromDL.predict(test_data, batch_size=batchsize, verbose=1)
validProbs = chromDL.predict(valid_data, batch_size=batchsize, verbose=1)

print("Predictions finished")

train_input = trainProbs[:, :, np.newaxis]
test_input = testProbs[:, :, np.newaxis]
valid_input = validProbs[:, :, np.newaxis]

print("New inputs Loaded")

# Enhancer model definition
enh_model = tf.keras.models.Sequential([
	tf.keras.layers.Convolution1D(64, 9, activation='relu', input_shape=(919,1), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3), kernel_constraint=max_norm(1)),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.LeakyReLU(0),
	tf.keras.layers.MaxPooling1D(9, 3),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Convolution1D(128, 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3), kernel_constraint=max_norm(1)),
	tf.keras.layers.LeakyReLU(0),
	tf.keras.layers.MaxPooling1D(4, 2),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Convolution1D(256, 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3),kernel_constraint=max_norm(1)),
	tf.keras.layers.MaxPooling1D(4, 3),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(180, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
])

enh_model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])

filepath = filename[:4] + "_run_out"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=2)

print("Fitting model...")
history = enh_model.fit(train_input, trainLabelMatrix, epochs=epoch_num, batch_size=batchsize, shuffle=True,
    validation_data=(valid_input, validLabelMatrix), callbacks=[checkpointer, earlystopper], verbose=2)

print("Testing model...", enh_model.evaluate(test_input, testLabelMatrix, batch_size=batchsize, verbose=2))
predProbs = enh_model.predict(test_input, batch_size=batchsize, verbose=2)

# Calculate performance metrics
auc = roc_auc_score(testLabelMatrix[:], predProbs[:])
auprc = average_precision_score(testLabelMatrix[:], predProbs[:])

print(f"Cell line: {filename[:4]}")
print(f"AUC: {auc}")
print(f"AUPRC: {auprc}")
