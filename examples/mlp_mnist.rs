extern crate ndarray;
extern crate autograd as ag;

use std::default::Default;
use std::time::{Duration, Instant};


// softmax regression with Adam optimizer for mnist.
// 0.92 test accuracy after 5 epochs,
// 0.26 sec/epoch on 2.7GHz Intel Core i5
//
// Run "./download_mnist.sh" beforehand if you don't have dataset

macro_rules! eval_with_time {
  ($x:expr) => {
    {
      let start = Instant::now();
      let result = $x;
      let end = start.elapsed();
      println!("{}.{:03} sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
      result
    }
  };
}

#[cfg(not(test))]
fn main()
{
    let ((x_train, y_train), (x_test, y_test)) = dataset::load();

    // -- graph def --
    let ref x = ag::placeholder(&[-1, 28 * 28]);
    let ref y = ag::placeholder(&[-1, 1]);
    let ref w = ag::variable(ag::init::glorot_uniform(&[28 * 28, 10]));
    let ref b = ag::variable(ag::init::zeros(&[1, 10]));
    let ref z = ag::matmul(x, w) + b;
    let ref loss = ag::sparse_softmax_cross_entropy(z, y);
    let ref grads = ag::gradients(loss, &[w, b], None);
    let ref predictions = ag::argmax(z, -1, true);
    let ref accuracy = ag::reduce_mean(&ag::equals(predictions, y), 0, false);

    // -- actual training --
    let mut optimizer = ag::sgd::optimizers::Adam { ..Default::default() };
    let batch_size = 100;
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size;

    for epoch in 0..5 {
        let perm = ag::init::permutation(num_samples).to_vec();
        for i in 0..num_batches {
            let indices = perm[i..i + batch_size].to_vec();
            let x_batch = x_train.select(ndarray::Axis(0), indices.as_slice());
            let y_batch = y_train.select(ndarray::Axis(0), indices.as_slice());
            let feed_dict = ag::Input::new().add(x, x_batch).add(y, y_batch);
            ag::sgd::apply_gradients(&mut optimizer, &[w, b], grads, feed_dict);
        }
        println!("finish epoch {}", epoch);
    }

    // -- test --
    let feed_dict = ag::Input::new().add(x, x_test).add(y, y_test);
    println!("test accuracy: {}", accuracy.eval_with_input(feed_dict));
}

pub mod dataset {
    extern crate ndarray;
    use std::fs::File;
    use std::io;
    use std::io::Read;
    use std::mem;
    use std::path::Path;
    use std::process;

    type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

    /// load mnist dataset as "ndarray" objects.
    ///
    /// labels are sparse (vertical vector).
    pub fn load() -> ((NdArray, NdArray), (NdArray, NdArray))
    {
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

    fn load_images<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize)
    {
        let ref mut buf_reader = io::BufReader::new(File::open(path).expect(
            "Please run ./download_mnist.sh beforehand",
        ));
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
        buf_reader.read_exact(buf.as_mut());
        let ret = buf.into_iter().map(|x| (x as f32) / 255.).collect();
        (ret, num_image)
    }

    fn load_labels<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize)
    {
        let ref mut buf_reader = io::BufReader::new(File::open(path).unwrap());
        let magic = u32::from_be(read_u32(buf_reader));
        if magic != 2049 {
            panic!("Invalid magic number. Got expect 2049, got {}", magic);
        }
        let num_label = u32::from_be(read_u32(buf_reader)) as usize;
        // read labels
        let mut buf: Vec<u8> = vec![0u8; num_label];
        buf_reader.read_exact(buf.as_mut());
        let ret: Vec<f32> = buf.into_iter().map(|x| x as f32).collect();
        (ret, num_label)
    }

    fn read_u32<T: Read>(reader: &mut T) -> u32
    {
        let mut buf: [u8; 4] = [0, 0, 0, 0];
        reader.read_exact(&mut buf);
        unsafe { mem::transmute(buf) }
    }
}
