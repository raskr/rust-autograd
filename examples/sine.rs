//! Learning the sine function
extern crate autograd as ag;

use ag::{ndarray_ext as nd, NdArrayView};
use ag::optimizers::sgd::SGD;
use ag::prelude::*;
use ag::rand::seq::SliceRandom;
use ag::tensor_ops as T;
use ag::tensor_ops::*;
use ag::{ndarray_ext as array, Context};
use ag::{optimizers, NdArray};
use ndarray::s;
use std::ops::Deref;
use std::time::Instant;
use ag::optimizers::adam::Adam;

// https://stackoverflow.com/questions/13897316/approximating-the-sine-function-with-a-neural-network
fn main() {
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    let mut env = ag::VariableEnvironment::new();

    let width = 100;
    let w1 = env.set(rng.glorot_uniform(&[1, width]));
    let w2 = env.set(rng.glorot_uniform(&[width, width]));
    let w3 = env.set(rng.glorot_uniform(&[width, 1]));
    let opt = Adam::default("adam", env.default_namespace().current_var_ids(), &mut env);

    let max_epoch = 5000;
    let batch_size = 100usize;

    for _ in 0..max_epoch {
        // let x_batch = rng.standard_uniform(&[batch_size, 1]) * 10.;
        // let y_batch = x_batch.map(|a| a.sin());

        env.run(|ctx| {
            let x = standard_uniform(&[batch_size, 1], ctx)
                .map(|x| x.map(|x| x * 10.));
            let t = x.map(|x| x.map(|x| x.sin()));

            // let x = ctx.placeholder("x", &[-1, 1]);
            // let t = ctx.placeholder("t", &[-1, 1]);
            let h1 = tanh(matmul(x, ctx.variable(w1)));
            let h2 = tanh(matmul(h1, ctx.variable(w2)));
            let y = matmul(h2, ctx.variable(w3));
            let l2_loss = reduce_mean(reduce_sum(square(y - t), &[-1], false), &[0], false).show();
            let ns = ctx.default_namespace();
            let (vars, grads) = optimizers::grad_helper(l2_loss, &ns);
            let update_op = opt.get_update_op(&vars, &grads, ctx);

            let results = ctx.evaluator()
                .push(update_op)
                // .feed(x, x_batch.view())
                // .feed(t, y_batch.view())
                .run();

            // unwrap update ops
            for res in results {
                res.as_ref().unwrap();
            }
        });
    }
}
