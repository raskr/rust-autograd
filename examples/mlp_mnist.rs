extern crate autograd as ag;
extern crate ndarray;

use ag::array;
use ag::optimizers::adam;
use ag::tensor::Variable;
use ag::Graph;
use ndarray::s;
use std::time::Instant;

type Tensor<'tensor, 'scope> = ag::Tensor<'tensor, 'scope, f32>;

// This is a softmax regression with Adam optimizer for mnist.
// 0.918 test accuracy after 3 epochs, 0.11 sec/epoch on 2.7GHz Intel Core i5
//
// First, run "./download_mnist.sh" beforehand if you don't have dataset and then run
// "cargo run --example mlp_mnist --release --features mkl" in `examples` directory.
//
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

fn inputs(g: &Graph<f32>) -> (Tensor, Tensor) {
    let x = g.placeholder(&[-1, 28 * 28]);
    let y = g.placeholder(&[-1, 1]);
    (x, y)
}

// Writing in define-by-run style for fun.
fn main() {
    let ((x_train, y_train), (x_test, y_test)) = dataset::load();

    let w_arr = array::shared(array::glorot_uniform(&[28 * 28, 10]));
    let b_arr = array::shared(array::zeros(&[1, 10]));
    let adam_state = adam::AdamState::new(&[&w_arr, &b_arr]);

    let max_epoch = 3;
    let batch_size = 200isize;
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size as usize;

    for epoch in 0..max_epoch {
        timeit!({
            ag::with(|g| {
                let w = g.variable(w_arr.clone());
                let b = g.variable(b_arr.clone());
                let (x, y) = inputs(g);
                let z = g.matmul(x, w) + b;
                let loss = g.sparse_softmax_cross_entropy(z, &y);
                let grads = &g.grad(&[&loss], &[w, b]);
                let update_ops: &[Tensor] =
                    &adam::Adam::default().compute_updates(&[w, b], grads, &adam_state, g);

                let perm = ag::ndarray_ext::permutation(num_batches) * batch_size as usize;
                for i in perm.into_iter() {
                    let i = *i as isize;
                    let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
                    let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();
                    g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]);
                }
                println!("finish epoch {}", epoch);
            });
        });
    }

    ag::with(|g| {
        let w = g.variable(w_arr.clone());
        let b = g.variable(b_arr.clone());
        let (x, y) = inputs(g);

        // -- test --
        let z = g.matmul(x, w) + b;
        let predictions = g.argmax(z, -1, true);
        let accuracy = g.reduce_mean(&g.equal(predictions, &y), &[0, 1], false);
        println!(
            "test accuracy: {:?}",
            accuracy.eval(&[x.given(x_test.view()), y.given(y_test.view())])
        );
    })
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
        let x_train = as_arr(ndarray::IxDyn(&[num_image_train, 28 * 28]), train_x).unwrap();
        let y_train = as_arr(ndarray::IxDyn(&[num_label_train, 1]), train_y).unwrap();
        let x_test = as_arr(ndarray::IxDyn(&[num_image_test, 28 * 28]), test_x).unwrap();
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
