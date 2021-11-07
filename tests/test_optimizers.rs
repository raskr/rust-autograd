extern crate autograd as ag;
extern crate ndarray;
use ag::optimizers::adam;
use ag::prelude::*;
use ag::tensor_ops as T;
use ag::variable::NamespaceTrait;
use ag::EvalError::OpError;
use ndarray::array;

type Tensor<'g> = ag::Tensor<'g, f32>;

#[test]
fn test_adam() {
    let mut ctx = ag::VariableEnvironment::<f32>::new();
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    let w = ctx
        .default_namespace_mut()
        .slot()
        .set(rng.glorot_uniform(&[2, 2]));
    let b = ctx
        .default_namespace_mut()
        .slot()
        .set(ag::ndarray_ext::zeros(&[1, 2]));

    // Prepare adam optimizer
    let adam = adam::Adam::default(
        "my_unique_adam",
        ctx.default_namespace().current_var_ids(),
        &mut ctx, // mut env
    );

    println!(
        "default ns: current_var_names: {:?}",
        ctx.default_namespace().current_var_names()
    );
    println!(
        "default ns: current_var_ids: {:?}",
        ctx.default_namespace().current_var_ids()
    );

    println!(
        "adam ns: current_var_names: {:?}",
        ctx.namespace("my_unique_adam").current_var_names()
    );
    println!(
        "adam ns: current_var_ids: {:?}",
        ctx.namespace("my_unique_adam").current_var_ids()
    );

    ctx.run(|g| {
        let x = T::convert_to_tensor(array![[0.1, 0.2], [0.2, 0.1]], g).show();
        let y = T::convert_to_tensor(array![1., 0.], g).show();
        let w = g.variable(w);
        let b = g.variable(b);
        let z = T::matmul(x, w) + b;
        let loss = T::sparse_softmax_cross_entropy(z, &y);
        let mean_loss = T::reduce_mean(loss, &[0], false);
        let grads = &T::grad(&[&mean_loss], &[w, b]);
        adam.update(&[w, b], grads, g, ag::Feeder::new());
    });
}
