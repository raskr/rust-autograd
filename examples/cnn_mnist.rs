//! Demonstration of MNIST digits classification with convolutional net.
//!
//! With accelerate, got 0.987 test accuracy in 70 sec on Apple M1 (8core CPU, 16GB RAM).
//!
//! First, run "./download_mnist.sh" beforehand if you don't have dataset and then run
//! "cargo run --example cnn_mnist --release --features blas,<blas-impl>".
use autograd as ag;
use ndarray;

use ag::optimizers;
use ag::prelude::*;
use ag::rand::seq::SliceRandom;
use ag::tensor_ops as T;
use ag::{ndarray_ext as array, Context};
use ndarray::s;
use std::time::Instant;

type Tensor<'graph> = ag::Tensor<'graph, f32>;

mod mnist_data;

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

fn conv_pool<'g>(x: Tensor<'g>, w: Tensor<'g>, b: Tensor<'g>, train: bool) -> Tensor<'g> {
    let y1 = T::conv2d(x, w, 1, 1) + b;
    let y2 = T::relu(y1);
    let y3 = T::max_pool2d(y2, 2, 0, 2);
    T::dropout(y3, 0.25, train)
}

fn compute_logits<'g>(c: &'g Context<f32>, train: bool) -> Tensor<'g> {
    let x = c.placeholder("x", &[-1, 28 * 28]);
    let x = x.reshape(&[-1, 1, 28, 28]); // 2D -> 4D
    let z1 = conv_pool(x, c.variable("w1"), c.variable("b1"), train); // map to 32 channel
    let z2 = conv_pool(z1, c.variable("w2"), c.variable("b2"), train); // map to 64 channel
    let z3 = T::reshape(z2, &[-1, 64 * 7 * 7]); // flatten
    let z4 = T::matmul(z3, c.variable("w3")) + c.variable("b3");
    T::dropout(z4, 0.25, train)
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut ag::ndarray_ext::get_default_rng());
    perm
}

fn main() {
    // Get training data
    let ((x_train, y_train), (x_test, y_test)) = mnist_data::load();

    let max_epoch = 5;
    let batch_size = 128isize;
    let num_train_samples = x_train.shape()[0];
    let num_batches = num_train_samples / batch_size as usize;

    // Create trainable variables in the default namespace
    let mut env = ag::VariableEnvironment::<f32>::new();
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    env.name("w1")
        .set(rng.random_normal(&[32, 1, 3, 3], 0., 0.1));
    env.name("w2")
        .set(rng.random_normal(&[64, 32, 3, 3], 0., 0.1));
    env.name("w3").set(rng.glorot_uniform(&[64 * 7 * 7, 10]));
    env.name("b1").set(array::zeros(&[1, 32, 28, 28]));
    env.name("b2").set(array::zeros(&[1, 64, 14, 14]));
    env.name("b3").set(array::zeros(&[1, 10]));

    // Prepare adam optimizer
    let adam = optimizers::Adam::default(
        "my_adam",
        env.default_namespace().current_var_ids(),
        &mut env,
    );

    // Training loop
    for epoch in 0..max_epoch {
        let mut loss_sum = 0f32;
        timeit!({
            for i in get_permutation(num_batches) {
                let i = i as isize * batch_size;
                let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
                let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();

                env.run(|ctx| {
                    let logits = compute_logits(ctx, true);
                    let loss =
                        T::sparse_softmax_cross_entropy(logits, ctx.placeholder("y", &[-1, 1]));
                    let mean_loss = T::reduce_mean(loss, &[0], false);
                    let ns = ctx.default_namespace();
                    let (vars, grads) = optimizers::grad_helper(&[mean_loss], &ns);
                    let update_op = adam.get_update_op(&vars, &grads, ctx);

                    let eval_results = ctx
                        .evaluator()
                        .push(mean_loss)
                        .push(update_op)
                        .feed("x", x_batch)
                        .feed("y", y_batch)
                        .run();

                    eval_results[1].as_ref().expect("parameter updates ok");
                    loss_sum += eval_results[0].as_ref().unwrap()[0];
                });
            }
            println!(
                "finish epoch {}, test loss: {}",
                epoch,
                loss_sum / num_batches as f32
            );
        });
    }

    // -- test --
    env.run(|ctx| {
        let logits = compute_logits(ctx, false);
        let predictions = T::argmax(logits, -1, true);
        let accuracy = T::reduce_mean(
            &T::equal(predictions, ctx.placeholder("y", &[-1, 1])),
            &[0, 1],
            false,
        );
        println!(
            "test accuracy: {:?}",
            ctx.evaluator()
                .push(accuracy)
                .feed("x", x_test.view())
                .feed("y", y_test.view())
                .run()
        );
    })
}
