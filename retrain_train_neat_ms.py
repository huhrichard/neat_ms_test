import NeatMS as ntms

raw_data_folder_path = './data/test_data/mzML/'
# Using peaks that have been aligned across samples
feature_table_path = './data/test_data/aligned_features.csv'
# Using unaligned peaks (One individual peak table for each sample)
# feature_table_path = '../data/test_data/unaligned_features/'
# This is important for NeatMS to read the feature table correctly
input_data = 'mzmine'
# input_data = 'xcms'

experiment = ntms.Experiment(raw_data_folder_path, feature_table_path, input_data)

for sample in experiment.samples:
    print('Sample {} : {} peaks'.format(sample.name,len(sample.feature_list)))

from  collections import Counter
exp = experiment
sizes = []
print("# Feature collection:",len(exp.feature_tables[0].feature_collection_list))

for consensus_feature in exp.feature_tables[0].feature_collection_list:
    sizes.append(len(consensus_feature.feature_list))

c = Counter(sizes)
print("Number of consensus features:")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list))
nn_handler = ntms.NN_handler(experiment)
print('test1')
nn_handler.create_batches(validation_split=0.1, normalise_class=False)
print('test2')
print('X train:', nn_handler.X_train)
print('y train:', nn_handler.y_train)

model_path = "./data/model/neatms_default_model.h5"
nn_handler.create_model(model = model_path)

layer_names = ['conv2d_1','conv2d_2','max_pooling2d_1']
for layer_name in layer_names:
    nn_handler.class_model.get_layer(layer_name).trainable = False

# Or using their position in the network (Here we freeze the first 4 layers)
# It is the same as above as the input is considered a layer with no parameters
for layer in nn_handler.class_model.layers[:4]:
    layer.trainable = False

# Here is how to make sure that the right layers are still trainable
for layer in nn_handler.class_model.layers:
    print(layer, layer.trainable)

from keras.optimizers import SGD, Adam
lr = 0.00001
opt = Adam(lr=lr)
nn_handler.class_model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy','mae'])

nn_handler.train_model()

# Set the threshold to 0.22
threshold=0.22
# Run the prediction
nn_handler.predict_peaks(threshold)



from  collections import Counter
exp = experiment
hq_sizes = []
lq_sizes = []
n_sizes = []
sizes = []
print("# Feature collection:",len(exp.feature_tables[0].feature_collection_list))
for consensus_feature in exp.feature_tables[0].feature_collection_list:
    hq_size = 0
    lq_size = 0
    n_size = 0
    for feature in consensus_feature.feature_list:
        for peak in feature.peak_list:
            if peak.valid:
                if peak.prediction.label == "High_quality":
                    hq_size += 1
                if peak.prediction.label == "Low_quality":
                    lq_size += 1
                if peak.prediction.label == "Noise":
                    n_size += 1

    hq_sizes.append(hq_size)
    lq_sizes.append(lq_size)
    n_sizes.append(n_size)
    sizes.append(len(consensus_feature.feature_list))

c = Counter(hq_sizes)
print("\nNumber of consensus features labeled as 'High quality':")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list))

c = Counter(lq_sizes)
print("\nNumber of consensus features labeled as 'Low quality':")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list))

c = Counter(n_sizes)
print("\nNumber of consensus features labeled as 'Noise':")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list))

filename = 'neatms_export.csv'

experiment.export_csv(filename)
# We create the dataframe using this function
NeatMS_output_df = experiment.export_to_dataframe()
# And display it
print(NeatMS_output_df)


# We add those specific properties to the export list
# Default properties will be overwritten, so make sure to add them to the list as well
export_properties = ["rt", "mz", "height", "area", "label", "peak_rt_start", "peak_rt_end"]

# Here is the full list of available properties that you can export
# ["rt", "mz", "height", "area", "label", "peak_rt_start", "peak_rt_end", "peak_mz_min", "peak_mz_max", "area_bc", "sn"]

NeatMS_output_df = experiment.export_to_dataframe(export_properties = export_properties)

print(NeatMS_output_df)

filename = 'neatms_export_with_extra_properties.csv'

experiment.export_csv(filename, export_properties = export_properties)


