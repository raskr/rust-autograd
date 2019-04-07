extern crate autograd as ag;
#[macro_use(s)]
extern crate ndarray;

use std::time::Instant;

type Tensor = ag::Tensor<f32>;

// This is a softmax regression with Adam optimizer for mnist.
// 0.918 test accuracy after 3 epochs, 0.14 sec/epoch on 2.7GHz Intel Core i5
//
// First, run "./download_mnist.sh" beforehand if you don't have dataset and then run
// "cargo run --example mlp_mnist --release" in `examples` directory.
//
// NOTE: This example is written in define-by-run style, so
// the performance is spoiled little bit.
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

fn logits(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    ag::matmul(x, w) + b
}

fn inputs() -> (Tensor, Tensor) {
    let x = ag::placeholder(&[-1, 28 * 28]);
    let y = ag::placeholder(&[-1, 1]);
    (x, y)
}

fn main() {
    let ((x_train, y_train), (x_test, y_test)) = dataset::load();

    // -- variable tensors (target of optimization) --
    let ref w = ag::variable(ag::ndarray_ext::glorot_uniform(&[28 * 28, 10]));
    let ref b = ag::variable(ag::ndarray_ext::zeros(&[1, 10]));
    let params = &[w, b];
    let ref params = ag::gradient_descent_ops::Adam::vars_with_states(params);

    // -- actual training --
    let max_epoch = 3;
    let batch_size = 200isize;
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size as usize;

    for epoch in 0..max_epoch {
        timeit!({
            let perm = ag::ndarray_ext::permutation(num_batches) * batch_size as usize;
            for i in perm.into_iter() {
                let (x, y) = inputs();
                let z = logits(&x, w, b);
                let loss = ag::sparse_softmax_cross_entropy(z, &y);
                let mean_loss = ag::reduce_mean(loss, &[0, 1], false);
                let grads = &ag::grad(&[&mean_loss], &[w, b]);
                let adam = ag::gradient_descent_ops::Adam::default();
                let update_ops: &[Tensor] = &adam.compute_updates(params, grads);

                let i = *i as isize;
                let x_batch = x_train.slice(s![i..i + batch_size, ..]).to_owned();
                let y_batch = y_train.slice(s![i..i + batch_size, ..]).to_owned();
                ag::eval(update_ops, &[(&x, &x_batch), (&y, &y_batch)]);
            }
        });
        println!("finish epoch {}", epoch);
    }

    // -- test --
    let (x, y) = inputs();
    let z = logits(&x, w, b);
    let predictions = ag::argmax(z, -1, true);
    let accuracy = ag::reduce_mean(&ag::equal(predictions, &y), &[0, 1], false);
    println!(
        "test accuracy: {:?}",
        accuracy.eval(&[(&x, &x_test), (&y, &y_test)])
    );
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
