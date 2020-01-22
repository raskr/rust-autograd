extern crate autograd as ag;
extern crate ndarray;

use ag::array;
use ag::optimizers::adam;
use ag::tensor::Variable;
use ag::Graph;
use ndarray::s;
use std::time::Instant;

type Tensor<'t, 's> = ag::Tensor<'t, 's, f32>;

// This is a toy convolutional network for MNIST.
// Got 0.987 test accuracy in 350 sec on 2.7GHz Intel Core i5.
//
// First, run "./download_mnist.sh" beforehand if you don't have dataset and then run
// "cargo run --example cnn_mnist --release --features mkl" in `examples` directory.
macro_rules! timeit {
    ($x:expr) => {{
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}

fn conv_pool<'t, 's: 't>(
    x: Tensor<'t, 's>,
    w: Tensor<'t, 's>,
    b: Tensor<'t, 's>,
) -> Tensor<'t, 's> {
    let g = x.graph;
    let y1 = g.conv2d(x, w, 1, 1) + b;
    let y2 = g.relu(y1);
    g.max_pool2d(y2, 2, 0, 2)
}

fn logits<'t, 's: 't>(x: Tensor<'t, 's>, w: Tensor<'t, 's>, b: Tensor<'t, 's>) -> Tensor<'t, 's> {
    x.graph.matmul(x, w) + b
}

fn inputs(g: &Graph<f32>) -> (Tensor, Tensor) {
    let x = g.placeholder(&[-1, 1, 28, 28]);
    let y = g.placeholder(&[-1, 1]);
    (x, y)
}

fn main() {
    let ((x_train, y_train), (x_test, y_test)) = dataset::load();

    let max_epoch = 5;
    let batch_size = 200isize;
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size as usize;

    let w1_ = array::shared(array::random_normal(&[32, 1, 3, 3], 0., 0.1));
    let w2_ = array::shared(array::random_normal(&[64, 32, 3, 3], 0., 0.1));
    let w3_ = array::shared(array::glorot_uniform(&[64 * 7 * 7, 10]));
    let b1_ = array::shared(array::zeros(&[1, 32, 28, 28]));
    let b2_ = array::shared(array::zeros(&[1, 64, 14, 14]));
    let b3_ = array::shared(array::zeros(&[1, 10]));
    let adam_state = adam::AdamState::new(&[&w1_, &w2_, &w3_, &b1_, &b2_, &b3_]);

    ag::with(|g| {
        let w1 = g.variable(w1_);
        let w2 = g.variable(w2_);
        let w3 = g.variable(w3_);
        let b1 = g.variable(b1_);
        let b2 = g.variable(b2_);
        let b3 = g.variable(b3_);
        let params = &[w1, w2, w3, b1, b2, b3];
        let (x, y) = inputs(g);
        let z1 = conv_pool(x, w1, b1); // map to 32 channel
        let z2 = conv_pool(z1, w2, b2); // map to 64 channel
        let z3 = g.reshape(z2, &[-1, 64 * 7 * 7]); // flatten
        let logits = logits(z3, w3, b3); // linear
        let loss = g.sparse_softmax_cross_entropy(&logits, &y);
        let grads = &g.grad(&[&loss], params);
        let update_ops: &[Tensor] =
            &adam::Adam::default().compute_updates(params, grads, &adam_state, g);

        for epoch in 0..max_epoch {
            timeit!({
                let perm = ag::ndarray_ext::permutation(num_batches) * batch_size as usize;
                for i in perm.into_iter() {
                    let i = *i as isize;
                    let x_batch = x_train.slice(s![i..i + batch_size, .., .., ..]).into_dyn();
                    let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();
                    g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]);
                }
            });
            println!("finish epoch {}", epoch);
        }

        // -- test --
        let predictions = g.argmax(logits, -1, true);
        let accuracy = g.reduce_mean(&g.equal(predictions, &y), &[0, 1], false);
        println!(
            "test accuracy: {:?}",
            accuracy.eval(&[x.given(x_test.view()), y.given(y_test.view())])
        );
    });
}

pub mod dataset {
    extern crate ndarray;
    use std::fs::File;
    use std::io;
    use std::io::Read;
    use std::mem;
    use std::path::Path;

    type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

    /// load mnist dataset as "ndarray" objects.
    ///
    /// labels are sparse (vertical vector).
    pub fn load() -> ((NdArray, NdArray), (NdArray, NdArray)) {
        // load dataset as `Vec`s
        let (train_x, num_image_train): (Vec<f32>, usize) =
            load_images("data/mnist/train-images-idx3-ubyte");
        let (train_y, num_label_train): (Vec<f32>, usize) =
            load_labels("data/mnist/train-labels-idx1-ubyte");
        let (test_x, num_image_test): (Vec<f32>, usize) =
            load_images("data/mnist/t10k-images-idx3-ubyte");
        let (test_y, num_label_test): (Vec<f32>, usize) =
            load_labels("data/mnist/t10k-labels-idx1-ubyte");

        // Vec to ndarray
        let as_arr = NdArray::from_shape_vec;
        let x_train = as_arr(ndarray::IxDyn(&[num_image_train, 1, 28, 28]), train_x).unwrap();
        let y_train = as_arr(ndarray::IxDyn(&[num_label_train, 1]), train_y).unwrap();
        let x_test = as_arr(ndarray::IxDyn(&[num_image_test, 1, 28, 28]), test_x).unwrap();
        let y_test = as_arr(ndarray::IxDyn(&[num_label_test, 1]), test_y).unwrap();
        ((x_train, y_train), (x_test, y_test))
    }

    fn load_images<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize) {
        let ref mut buf_reader = io::BufReader::new(
            File::open(path).expect("Please run ./download_mnist.sh beforehand"),
        );
        let magic = u32::from_be(read_u32(buf_reader));
        if magic != 2051 {
            panic!("Invalid magic number. expected 2051, got {}", magic)
        }
        let num_image = u32::from_be(read_u32(buf_reader)) as usize;
        let rows = u32::from_be(read_u32(buf_reader)) as usize;
        let cols = u32::from_be(read_u32(buf_reader)) as usize;
        assert!(rows == 28 && cols == 28);

        // read images
        let mut buf: Vec<u8> = vec![0u8; num_image * rows * cols];
        let _ = buf_reader.read_exact(buf.as_mut());
        let ret = buf.into_iter().map(|x| (x as f32) / 255.).collect();
        (ret, num_image)
    }

    fn load_labels<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize) {
        let ref mut buf_reader = io::BufReader::new(File::open(path).unwrap());
        let magic = u32::from_be(read_u32(buf_reader));
        if magic != 2049 {
            panic!("Invalid magic number. Got expect 2049, got {}", magic);
        }
        let num_label = u32::from_be(read_u32(buf_reader)) as usize;
        // read labels
        let mut buf: Vec<u8> = vec![0u8; num_label];
        let _ = buf_reader.read_exact(buf.as_mut());
        let ret: Vec<f32> = buf.into_iter().map(|x| x as f32).collect();
        (ret, num_label)
    }

    fn read_u32<T: Read>(reader: &mut T) -> u32 {
        let mut buf: [u8; 4] = [0, 0, 0, 0];
        let _ = reader.read_exact(&mut buf);
        unsafe { mem::transmute(buf) }
    }
}
