import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_img(img, mask=None):
    plt.imshow(img, cmap='bone')

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(1.0,0.0,0.0)]]
        labels = ["Dust spot"]
        plt.legend(handles,labels)
    plt.axis('off')


def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5*5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy()*255.0
        msk = msk.astype('uint8')
        show_img(img, msk)
    plt.tight_layout()
    plt.show()