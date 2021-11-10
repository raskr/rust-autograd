//! This is a softmax regression with Adam optimizer for mnist.
//! 0.918 test accuracy after 3 epochs, 0.11 sec/epoch on 2.7GHz Intel Core i5
//!
//! First, run "./download_mnist.sh" beforehand if you don't have dataset and then run
//! "cargo run --example mlp_mnist --release --features mkl" in `examples` directory.
use autograd as ag;
use ndarray;


use ag::optimizers::Adam;
use ag::prelude::*;
use ag::rand::seq::SliceRandom;
use ag::tensor_ops as T;

use ag::{ndarray_ext as array, Context};
use ndarray::s;
use std::ops::Deref;
use std::time::Instant;

mod mnist_data;

type Tensor<'graph> = ag::Tensor<'graph, f32>;

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

fn inputs<'g>(g: &'g Context<f32>) -> (Tensor<'g>, Tensor<'g>) {
    let x = g.placeholder("x", &[-1, 28 * 28]);
    let y = g.placeholder("y", &[-1, 1]);
    (x, y)
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut rand::thread_rng());
    perm
}

// Writing in define-by-run style for fun.
fn main() {
    let ((x_train, y_train), (x_test, y_test)) = mnist_data::load();

    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();

    let mut env = ag::VariableEnvironment::new();
    let w = env.set(rng.glorot_uniform(&[28 * 28, 10]));
    let b = env.set(array::zeros(&[1, 10]));

    let adam = Adam::default("adam", env.default_namespace().current_var_ids(), &mut env);

    let max_epoch = 3;
    let batch_size = 200isize;
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size as usize;

    for epoch in 0..max_epoch {
        timeit!({
            for i in get_permutation(num_batches) {
                let i = i as isize * batch_size;
                let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
                let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();

                env.run(|c| {
                    let w = c.variable(w);
                    let b = c.variable(b);
                    let (x, y) = inputs(c.deref());
                    let z = T::matmul(x, w) + b;
                    let loss = T::sparse_softmax_cross_entropy(z, &y);
                    let mean_loss = T::reduce_mean(loss, &[0], false);
                    let grads = &T::grad(&[&mean_loss], &[w, b]);

                    let mut feeder = ag::Feeder::new();
                    feeder.push(x, x_batch).push(y, y_batch);
                    adam.update(&[w, b], grads, c, feeder);
                });
            }
        });
        println!("finish epoch {}", epoch);
    }

    env.run(|c| {
        let w = c.variable(w);
        let b = c.variable(b);
        let (x, y) = inputs(c.deref());

        // -- test --
        let z = T::matmul(x, w) + b;
        let predictions = T::argmax(z, -1, true);
        let accuracy = T::reduce_mean(&T::equal(predictions, &y), &[0, 1], false);
        println!(
            "test accuracy: {:?}",
            c.evaluator()
                .push(accuracy)
                .feed(x, x_test.view())
                .feed(y, y_test.view())
                .run()
        );
    })
}
