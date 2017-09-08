extern crate ndarray;
extern crate autograd as ag;

use std::time::{Duration, Instant};
use std::default::Default;


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

fn main() {
    let ((x_train, y_train), (x_test, y_test)) = ag::dataset::mnist::load();

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
    let mut optimizer = ag::sgd::Adam { ..Default::default() };
//     let mut optimizer = ag::sgd::SGD { lr: 0.1 };
    let batch_size = 100;
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size;

    for epoch in 0..5 {
//                eval_with_time!({
        let perm = ag::init::permutation(num_samples).to_vec();
        for i in 0..num_batches {
            let indices = perm[i..i+batch_size].to_vec();
            let x_batch = x_train.select(ndarray::Axis(0), indices.as_slice());
            let y_batch = y_train.select(ndarray::Axis(0), indices.as_slice());
            let feed_dict = ag::Input::new().add(x, x_batch).add(y, y_batch);
            ag::train::apply_gradients(&mut optimizer, &[w, b], grads, feed_dict);
        }
//        });
        println!("finish epoch {}", epoch);
    }

    // -- test --
    let feed_dict = ag::Input::new().add(x, x_test).add(y, y_test);
    println!("test accuracy: {}", accuracy.eval_with_input(feed_dict));
}
