import swanlab
import time
import random

lr = 0.01
epochs = 20
offset = random.random() / 5

swanlab.init(
    # Set experiment name
    experiment_name="sample_experiment",
    # Set description
    description="This is a sample experiment for machine learning training.",
    # Record tracked hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
    logdir="./logs"
)

# Simulated machine learning training process
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # Tracking index: 'loss' and 'accuracy'
    swanlab.log({"loss": loss, "accuracy": acc})
    time.sleep(1)