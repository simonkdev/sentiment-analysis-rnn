from main_rnn import RNN

rnn = RNN()
#rnn.load_trained_state()
#rnn.calculate_accuracy()
#rnn.backprop.save_state()
rnn.train(2000)
rnn.backprop.save_state()
rnn.calculate_accuracy()
