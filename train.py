from main_rnn import RNN

TOTAL_ITERATIONS = 2000
CHECKPOINT_INTERVAL = 50
BATCH_SIZE = 256

rnn = RNN(load_dataset=True)

for completed_iterations in range(0, TOTAL_ITERATIONS, CHECKPOINT_INTERVAL):
    iterations = min(CHECKPOINT_INTERVAL, TOTAL_ITERATIONS - completed_iterations)
    next_checkpoint = completed_iterations + iterations

    print(f"\n[ TRAIN ] Running iterations {completed_iterations + 1}-{next_checkpoint}...")
    rnn.train(iterations=iterations, batch_size=BATCH_SIZE)

    print(f"[ EVAL ] Accuracy after {next_checkpoint} iterations:")
    rnn.calculate_accuracy()

    print(f"[ SAVE ] Saving state after {next_checkpoint} iterations...")
    rnn.save_trained_state()

print("[ DONE ] Training complete.")
