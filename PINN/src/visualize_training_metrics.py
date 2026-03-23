import torch
from matplotlib import pyplot as plt



def plot_l2(steps, l2_error, l2_error_name):
    # Plot L2
    print(f"Saving: {l2_error_name}.png")
    plt.figure(figsize=(10, 5))
    plt.semilogy(steps, l2_error)
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.title('l2 error')
    plt.grid(True)
    plt.savefig(f'{l2_error_name}.png', dpi=150)

def plot_loss(loss, loss_name):
    # Plot training loss
    print(f"Saving: {loss_name}.png")
    plt.figure(figsize=(10, 5))
    plt.semilogy(loss)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(f'{loss_name}.png', dpi=150)



import sys
if __name__ == "__main__":
    if len(sys.argv) > 1:
        dir_name = sys.argv[1]
    else:
        dir_name = 'run_latest'
    print(f"Will be working in directory '{dir_name}'...")

    l2_error_name = f'{dir_name}/training_l2_error'
    loss_name = f'{dir_name}/training_loss'
    print(f"Loading: {l2_error_name}.pth")
    print(f"Loading: {loss_name}.pth")
    l2_error = torch.load(f'{l2_error_name}.pth')
    loss = torch.load(f'{loss_name}.pth')

    import utility
    model_metadata = utility.json_load(f"{dir_name}/model_metadata.json")
    n_steps_log = model_metadata["args"]["testing_frequency"]
    n_logged_pnts = len(l2_error)
    steps = n_steps_log*torch.linspace(1,n_logged_pnts,n_logged_pnts, dtype=torch.int)

    plot_l2(steps, l2_error, l2_error_name)
    plot_loss(loss, loss_name)
