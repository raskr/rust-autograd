extern crate autograd as ag;
extern crate ndarray;
use ag::prelude::*;
use ag::tensor_ops as T;
use ag::variable::NamespaceTrait;
use ag::{optimizers, VariableEnvironment};

use ndarray::array;

#[test]
fn test_adam() {
    let mut env = make_env();
    let opt = optimizers::Adam::default(
        "my_unique_adam",
        env.default_namespace().current_var_ids(),
        &mut env, // mut env
    );
    run(opt, env);
}

#[test]
fn test_adagrad() {
    let mut env = make_env();
    let opt = optimizers::AdaGrad::default(
        "my_unique_adagrad",
        env.default_namespace().current_var_ids(),
        &mut env, // mut env
    );
    run(opt, env);
}

#[test]
fn test_momentum() {
    let mut env = make_env();
    let opt = optimizers::MomentumSGD::default(
        "my_momentum_sgd",
        env.default_namespace().current_var_ids(),
        &mut env, // mut env
    );
    run(opt, env);
}

fn make_env() -> VariableEnvironment<f64> {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    env.name("w").set(rng.glorot_uniform(&[2, 2]));
    env.name("b").set(ag::ndarray_ext::zeros(&[1, 2]));
    env
}

fn run<O: Optimizer<f64>>(opt: O, env: VariableEnvironment<f64>) {
    env.run(|g| {
        let x = T::convert_to_tensor(array![[0.1, 0.2], [0.2, 0.1]], g).show();
        let y = T::convert_to_tensor(array![1., 0.], g).show();
        let w = g.variable("w");
        let b = g.variable("b");
        let z = T::matmul(x, w) + b;
        let loss = T::sparse_softmax_cross_entropy(z, &y);
        let mean_loss = T::reduce_mean(loss, &[0], false);
        let ns = g.default_namespace();
        let (vars, grads) = optimizers::grad_helper(&[mean_loss], &ns);
        opt.update(&vars, &grads, g, ag::Feeder::new());
    });
}
