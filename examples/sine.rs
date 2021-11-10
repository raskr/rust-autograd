//! Learning the sine function
extern crate autograd as ag;

use ag::prelude::*;
use ag::tensor_ops::*;
use ag::optimizers;
use ag::optimizers::adam::Adam;

fn main() {
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    let mut env = ag::VariableEnvironment::new();

    let width = 64;
    let w1 = env.set(rng.glorot_uniform(&[1, width]));
    let w2 = env.set(rng.glorot_uniform(&[width, width]));
    let w3 = env.set(rng.glorot_uniform(&[width, 1]));
    let opt = Adam::default("adam", env.default_namespace().current_var_ids(), &mut env);

    let max_epoch = 500;
    let batch_size = 50usize;

    for _ in 0..max_epoch {
        env.run(|ctx| {
            // training data
            let x = standard_uniform(&[batch_size, 1], ctx)
                .map(|x| x.map(|x| x * 10.));
            let t = x.map(|x| x.map(|x| x.sin()));

            // run nn
            let h1 = tanh(matmul(x, ctx.variable(w1)));
            let h2 = tanh(matmul(h1, ctx.variable(w2)));
            let y = matmul(h2, ctx.variable(w3));
            let mse = mean_all(square(y - t)); // mean squared error
            let ns = ctx.default_namespace();
            let (vars, grads) = optimizers::grad_helper(&[mse], &ns);
            let update_op = opt.get_update_op(&vars, &grads, ctx);

            let results = ctx.evaluator()
                .push(mse)
                .push(update_op)
                .run();

            println!("training loss: {}", results[0].as_ref().unwrap());
            results[1].as_ref().unwrap(); // update op
        });
    }
}
