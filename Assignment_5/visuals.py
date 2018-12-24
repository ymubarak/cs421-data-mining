import matplotlib.pyplot as plt


def plot_img_with_gnd(image, grnd_truth):
    fig = plt.figure(figsize=(50,50))
    num_columns = grnd_truth.shape[1] +1
    fig.add_subplot(1, num_columns, 1)
    plt.imshow(image)
    plt.axis('off')
    for i in range(num_columns-1):
        ax = fig.add_subplot(1, num_columns, i+2)
        grnd = grnd_truth[:,i][0][0][0][0]
        plt.imshow(grnd, cmap='Greys_r')
        plt.axis('off')


def plot_images_truths(images, ground_truth):
    for i in range(len(images)):
        plot_img_with_gnd(images[i], ground_truth[i]['groundTruth'])
