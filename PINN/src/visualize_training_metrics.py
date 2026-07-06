import torch
from matplotlib import pyplot as plt


def _to_plot_series(data):
    """Convert tensors to CPU arrays so matplotlib never sees device-backed values."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data



def plot_time_adapt_sampling(Ts, file_name):
    print(f"Saving: {file_name}.png")
    plt.figure(figsize=(10, 5))
    plt.plot(_to_plot_series(Ts))
    plt.xlabel('epochs')
    plt.ylabel('T (terminal time)')
    plt.title('Time adaptive sampling')
    plt.grid(True)
    plt.savefig(f'{file_name}.png', dpi=150)



def plot_hist_data_dict(hist_data_dict, file_name, label_fn, line_width_fn, color_fn, y_label, title, x=None):
    """
    hist_data_dict: ex. {'pde': [], 'bc': []}
    """
    print(f"Saving: {file_name}.png")
    plt.figure(figsize=(10, 5))
    
    if isinstance(hist_data_dict, dict):
        # plasma good for causal weights w_m
        #default_color_names = [
        #    name for name, series in hist_data_dict.items()
        #    if len(series) != 0 and color_fn(name) is None
        #]
        #if default_color_names:
        #    # Use a narrow slice of a sequential colormap so the lines stay visually related.
        #    plt.gca().set_prop_cycle(
        #        color=plt.cm.plasma(torch.linspace(0.2, 0.8, len(default_color_names)))
        #    )
        for name, series in hist_data_dict.items():
            if len(series) == 0:
                continue
            series = _to_plot_series(series)
            params = {
                "label": label_fn(name),
                "linewidth": line_width_fn(name)
            }
            if color_fn(name) != None:
                plt.semilogy(series, **params, color=color_fn(name))
            else:
                plt.semilogy(series, **params)
        
        # Increased legend font size
        plt.legend(fontsize=14)
    else:
        plt.semilogy(_to_plot_series(hist_data_dict))
    
    # Increased label and title font sizes
    plt.xlabel('epochs', fontsize=14)
    #plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    
    # Increased axis tick label sizes (numbers on X and Y axes)
    plt.tick_params(axis='both', labelsize=12)
    
    plt.grid(True)
    plt.savefig(f'{file_name}.png', dpi=150, bbox_inches='tight') # bbox_inches='tight' ensures labels aren't cut off


# term_type: pde, bc
# wl_type: weights, losses
def plot_causal_weights_losses(hist_data_dict, file_name, wl_type, term_type, t_discr):
    d = {}
    for i in range(len(t_discr)-1):
        d[f"{i+1}"] = rf"$, \:[t_{i}, t_{i+1}) = [{t_discr[i]}, {t_discr[i+1]})$"
    if wl_type == 'losses':
        linewidth_fn = lambda name: 1.0 if name == term_type else 1.0
        label_fn = lambda name: fr"$\ell^{{({name})}}$"+d[name] if name != term_type else rf"$\ell_{{{term_type}}}$"
        color_fn = lambda name: 'black' if name == term_type else None
    elif wl_type == 'weights':
        linewidth_fn = lambda name: 1.0
        label_fn = lambda name: fr"$w_{name}$"+d[name]
        color_fn = lambda name: None
    plot_hist_data_dict(hist_data_dict, file_name, label_fn, linewidth_fn, color_fn, wl_type, f"Causal Weighting: {term_type} {wl_type}")



def plot_lambda_adapt_weights(weights_data_dict, file_name, title):
    plot_hist_data_dict(weights_data_dict, file_name, lambda name: rf"$\lambda_{{{name}}}$", lambda name: 1.0, lambda name: None, 'weights', title)

def plot_loss(loss_data_dict, file_name):
    plot_hist_data_dict(loss_data_dict, file_name, lambda name: rf"$\ell_{{{name}}}$", lambda name: 1.0 if name == 'total' else 1.0, lambda name: 'black' if name == 'total' else None, 'loss', "Training loss")

def plot_mem(mem_data_dict, file_name):
    mem_series = {k: v for k, v in mem_data_dict.items() if k != "step"}
    plot_hist_data_dict(mem_series, file_name, lambda name: name, lambda name: 1.0, lambda name: None, 'MB', "Memory [MB]")

def plot_test_rel_l2(test_data_dict, file_name):
    plot_hist_data_dict(test_data_dict, file_name, lambda name: rf"rel $L^2_{{{name}}}$", lambda name: 1.0, lambda name: None, r'rel $L^2$ error', r"Testing: Relative $L^2$ error")
def plot_test_linf(test_data_dict, file_name):
    plot_hist_data_dict(test_data_dict, file_name, lambda name: rf"$L^\inf_{{{name}}}$", lambda name: 1.0, lambda name: None, r'$L^\inf$ error', r"Testing: $L^\inf$ error")



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

    plot_loss(loss, loss_name)
