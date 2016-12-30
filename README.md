## Image_retrieval
An image retrieval system based on bag-of-visual-word(BoW/BoF). 
As for image feature, SIFT descriptors is used.

## Requirements
I use `k-means` clustering from [`scipy`](https://www.scipy.org/). Make sure you had installed [`scipy`](https://www.scipy.org/) successfully.

## Hierarchical k-means
In this repository, we use hierarchical k-means instead of k-means to reduce the time for creating codebook. `Hierarchical k-means` in this repository was adapted from [`hierarchical kmeans`](https://github.com/github-pengge/hierarchical_kmeans).

## Usage
#### Step 1: Read images and compute feature of each images
For example, we can read the first 500 images in [`ukbench`](http://vis.uky.edu/~stewe/ukbench/) and compute their SIFT feature by
``` python
from utils import load_dataset
from feature import *
images = load_dataset('/path/to/ukbench', 'ukbench', first_n=500) # read first 500 images in ukbench
descriptors = sift_detect_and_compute(images, normalize=True, keep_top_k=500) # compute sift and keep 500 top-response descriptors of each images(if 500 descriptors are available)
```
You  can use other image feature like HSV color histogram, or SURF/LBP/Brisk, etc.

#### Step 2: Create codebook/vocabulary
To create codebook, we first have to determine how to do the hierarchical k-means, here, we will create a tree to tell how to do the k-means.
``` python
from hierarchical_kmeans import tree
clusters = tree('my_VT_structure')
for i in range(10):
    x = tree('l1-c%d' % (i+1))
    clusters.add_child(x)
    for j in range(10):
        y = tree('l2-c%d-c%d' % (i+1, j+1))
        x.add_child(y)
        for k in range(5):
            z = tree('l3-c%d-c%d-c%d' % (i+1, j+1, k+1))
            y.add_child(z)
```
This code create a tree with 3 levels, it tells the constructer to create a k-means tree with depth-3: for first level, doing a 10-class k-means; for each branch(10 branches) in level 2, doing a 10-class k-means; and for each branch in level 3, doing a 5-class k-means. Note that if you need a 100-class-clustering, k-means will not always get the exactly 100 classes, some class may be empty, so commonly speaking, the example given above will get a total clustering of less than `10*10*5` classes, but I promise this is not a problem.

Let's build a vocabulary tree! Try `subsampling=10`, only 1/10 descriptors are used for clustering, if your machine suffers from computing speed and memory cost. 

``` python
from vocabulary import construct_vocabulary
voc = construct_vocabulary(descriptors, clusters, 'my_voc_name', subsampling=-1)
```

#### Step 3: Doing query
Once vocabulary tree was built, we can find similar images of a certain query.
``` python
image = cv2.imread(im_name)
desciptor = sift_detect_and_compute([image], normalize=True, keep_top_k=500) 
response = voc.sims(descriptor)  # response is a list of image ids of training images.
```
You can achieve a recall@4 of 2.80 on [`ukbench`](http://vis.uky.edu/~stewe/ukbench/) if you build your codebook with the following hierarchical k-means structure:
``` python
from hierarchical_kmeans import tree
clusters = tree('my_VT_structure')
for i in range(100):
    x = tree('l1-c%d' % (i+1))
    clusters.add_child(x)
    for j in range(50):
        y = tree('l2-c%d-c%d' % (i+1, j+1))
        x.add_child(y)
        for k in range(20):
            z = tree('l3-c%d-c%d-c%d' % (i+1, j+1, k+1))
            y.add_child(z)
```

## Further improve performance
BoF is in fact a weak feature for image retrieval. You can combine several features to get better result. Also, fusion of several results can usually get surprising result.
For example, we had do the experiments, and found that if we fused results of BoF and deep feature matching, we could achieve recall@4 of 3.76 on [`ukbench`](http://vis.uky.edu/~stewe/ukbench/).
