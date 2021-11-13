#!/usr/bin/env bash

mkdir -p /tmp/rust-autograd/data/mnist
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output "/tmp/rust-autograd/data/mnist/train-images-idx3-ubyte.gz"
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output "/tmp/rust-autograd/data/mnist/train-labels-idx1-ubyte.gz"
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --output "/tmp/rust-autograd/data/mnist/t10k-images-idx3-ubyte.gz"
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --output "/tmp/rust-autograd/data/mnist/t10k-labels-idx1-ubyte.gz"
gzip -d /tmp/rust-autograd/data/mnist/*.gz