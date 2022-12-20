from matplotlib import pyplot as plt


def get_losses_graph(x, y:list, labels=[], savename=''):
    plt.figure(dpi=300, figsize=(24, 8))

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    _handles = []

    for i, _y in enumerate(y):
        ret, = plt.plot(x, _y, color[i], label=labels[i])
        _handles.append(ret)

    plt.xlabel('Epochs')
    plt.ylabel('Losses')

    plt.legend(handles=_handles, bbox_to_anchor=(0, -0.3), loc='lower left')

    plt.title("Loss Graph")
    plt.savefig(f"losses_{savename}.png" if savename != '' else "losses.png", format="png", bbox_inches="tight")
