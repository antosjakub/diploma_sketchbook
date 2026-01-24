import torch
from matplotlib import pyplot as plt


try:
    l2_error = torch.load('training_l2_error.pth')
    loss = torch.load('training_loss.pth')

    import json
    with open("args.json", "r") as f:
        metadata = json.load(f)
    n_steps_log = metadata["n_steps_log"]
    n_logged_pnts = len(l2_error)
    steps = n_steps_log*torch.linspace(1,n_logged_pnts,n_logged_pnts, dtype=torch.int)
except:
    print("Something went wrong with loading the files.")


# Plot L2
plt.figure(figsize=(10, 5))
plt.semilogy(steps, l2_error)
plt.xlabel('Step')
plt.ylabel('Error')
plt.title('l2 error')
plt.grid(True)
plt.savefig('training_l2_error.png', dpi=150)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.semilogy(loss)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('training_loss.png', dpi=150)
