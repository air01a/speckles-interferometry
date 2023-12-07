from matplotlib import pyplot as plt
import numpy as np

def show_image_3d(image,st=None):

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')


    # Utiliser la valeur des pixels comme coordonnées Z
    z = image

    # Afficher la surface
    ax.plot_surface(x, y, z)

    # Définir les libellés des axes
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Valeur des pixels')
    ax.view_init(56, 134, 0)
    # Afficher le graphique
    if st:
        st.pyplot(fig)
    else:
        plt.show()


def show_image(image,title="", st=None, cmap="viridis"):
    if st:
        
        fig, ax = plt.subplots()
        ax.imshow(image,cmap=cmap)
        ax.set_title(title)
        st.pyplot(fig)

    else:
        plt.figure()
        plt.imshow(image,cmap=cmap)
        plt.title(title)
        plt.show()    


def show_images(images, title="", cmap="viridis", column=5,max_images=30, st=None):

    column = min(column, len(images))
    max = min(max_images,len(images))
    fig, axs = plt.subplots(max//column, column)
    #fig.set_figheight(int(max/column)*8)
    #fig.set_figwidth(column*8)

    for i in range(0,max):
        axs[i//column,i%column].imshow(images[i])
    if st:
        st.pyplot(fig)
    else:
        fig.show()
