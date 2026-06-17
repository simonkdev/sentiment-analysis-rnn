from main_rnn import RNN

rnn = RNN(load_dataset=True)
rnn.train(iterations=20000, batch_size=256)
rnn.backprop.save_state()
rnn.calculate_accuracy()
