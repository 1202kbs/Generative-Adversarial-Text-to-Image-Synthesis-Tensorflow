import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(samples, shape):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
           ax = plt.subplot(gs[i])
           plt.axis('off')
           ax.set_xticklabels([])
           ax.set_yticklabels([])
           ax.set_aspect('equal')
           plt.imshow(sample.reshape(shape), cmap=plt.get_cmap('gray'))

        return fig