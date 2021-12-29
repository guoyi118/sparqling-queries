import pickle

pickle_in = open("model_output_2_beam.pickle","rb")
model_output_2_beam = pickle.load(pickle_in)
print(len(model_output_2_beam))
# print(list(model_output_2_beam.keys()))

print(model_output_2_beam['SPIDER_dev_1031']['beam_2rd_output'])

# print(model_output_2_beam['SPIDER_dev_0'])


# pickle_in = open("wrong_output.pickle","rb")
# wrong_output = pickle.load(pickle_in)
# print(len(wrong_output))
# print(list(wrong_output.keys()))

# print(wrong_output['SPIDER_dev_5'])


