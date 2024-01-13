import swanlab
import time

# Initialize
swanlab.init(logdir="./logs")

for epoch in range(1, 20):
    print("epoch", epoch)
    # Tracking index: `epoch`
    swanlab.log({"epoch": epoch})
    time.sleep(1)