extern crate autograd as ag;
extern crate ndarray;

use ag::ndarray_ext as array;
use ag::optimizers::adam;
use ag::rand::seq::SliceRandom;
use ag::tensor::Variable;
use ag::Graph;
use ndarray::s;
use std::time::Instant;

mod mnist_data;

type Tensor<'graph> = ag::Tensor<'graph, f32>;

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

// e.g. [4, 0, 2, 1, 3], given size = 5
fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut rand::thread_rng());
    perm
}

// Writing in define-by-run style for fun.
fn main() {
    let ((x_train, y_train), (x_test, y_test)) = mnist_data::load();

    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    let w_arr = array::into_shared(rng.glorot_uniform(&[28 * 28, 10]));
    let b_arr = array::into_shared(array::zeros(&[1, 10]));
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
                let mean_loss = g.reduce_mean(loss, &[0], false);
                let grads = &g.grad(&[&mean_loss], &[w, b]);
                let update_ops: &[Tensor] =
                    &adam::Adam::default().compute_updates(&[w, b], grads, &adam_state, g);

                for i in get_permutation(num_batches) {
                    let i = i as isize * batch_size;
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
