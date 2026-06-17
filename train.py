from main_rnn import RNN

TOTAL_ITERATIONS = 2000
CHECKPOINT_INTERVAL = 50
BATCH_SIZE = 256

rnn = RNN(load_dataset=True)
best_accuracy = float("-inf")

for completed_iterations in range(0, TOTAL_ITERATIONS, CHECKPOINT_INTERVAL):
    iterations = min(CHECKPOINT_INTERVAL, TOTAL_ITERATIONS - completed_iterations)
    next_checkpoint = completed_iterations + iterations

    print(f"\n[ TRAIN ] Running iterations {completed_iterations + 1}-{next_checkpoint}...")
    rnn.train(iterations=iterations, batch_size=BATCH_SIZE)

    print(f"[ EVAL ] Accuracy after {next_checkpoint} iterations:")
    accuracy = rnn.calculate_accuracy()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"[ SAVE ] New best accuracy: {best_accuracy:.2f}%. Saving state...")
        rnn.save_trained_state()
    else:
        print(f"[ SKIP ] Accuracy did not beat best score: {best_accuracy:.2f}%.")

print(f"[ DONE ] Training complete. Best accuracy: {best_accuracy:.2f}%.")
