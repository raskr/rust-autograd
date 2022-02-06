extern crate autograd as ag;
extern crate ndarray;

use ag::prelude::*;
use ag::tensor_ops as T;
use std::collections::HashMap;

#[test]
fn get() {
    let mut env = ag::VariableEnvironment::new();
    let v = &env.slot().set(ndarray::arr1(&[1., 2., 3.]));

    env.run(|graph| {
        let var = graph
            .var_tensors_by_id(graph.env())
            .collect::<HashMap<_, _>>();
        let v = var[v];
        let a: ag::Tensor<f64> = 2. * v;
        let z = a.access_elem(1);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn add_n() {
    let mut ctx = ag::VariableEnvironment::new();
    let v1 = ctx.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v2 = ctx.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v3 = ctx.slot().set(ndarray::arr1(&[1., 2., 3.]));
    ctx.run(|graph| {
        let v1 = graph.variable(v1);
        let v2 = graph.variable(v2);
        let v3 = graph.variable(v3);
        let z = T::add_n(&[v1, v2, v3]);
        let g = T::grad(&[z], &[v2]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v2],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn clip() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::clip(v, 1.5, 2.5);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn asinh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::asinh(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn acosh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::acosh(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn atanh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::atanh(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn sinh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::sinh(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn cosh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::cosh(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn tanh() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::tanh(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn asin() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::asin(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
fn acos() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::acos(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn atan() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::atan(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn sin() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::sin(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn cos() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::cos(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn tan() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0., 0.2));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::tan(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
fn pow() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0.9, 1.1));
    env.run(|ctx| {
        let v = ctx.variable(v);
        let z = T::pow(v, 1.1);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            ctx,
        );
    });
}

#[test]
fn sqrt() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0.9, 1.1));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::sqrt(v);
        T::add(v, z);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn exp() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 0.9, 1.1));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::exp(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
fn ln() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[3], 1., 1.1));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::ln(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
fn expand_dims() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::expand_dims(v, &[0, 2]);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn squeeze() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 1, 2, 1]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::squeeze(v, &[3, 1]);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn matmul() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let a = T::convert_to_tensor(rng.standard_normal(&[4, 2]), graph);
        let v = graph.variable(v);
        let z = T::matmul(a, v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            5e-3,
            graph,
        );
    });
}

#[test]
fn batch_matmul() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2, 3]));
    env.run(|graph| {
        let a = T::convert_to_tensor(rng.standard_normal(&[2, 4, 2]), graph);
        let v = graph.variable(v);
        let z = T::batch_matmul(a, v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn implicit_broadcast() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let b = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let x = T::convert_to_tensor(rng.standard_normal(&[4, 3]), graph);
        let b = graph.variable(b);
        let z = x + b;
        let g = T::grad(&[z], &[b]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[b],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn wx_plus_b() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let w = env.slot().set(rng.standard_normal(&[2, 3]));
    let b = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let x = T::convert_to_tensor(rng.standard_normal(&[4, 2]), graph);
        let w = graph.variable(w);
        let b = graph.variable(b);
        let z = T::matmul(x, w) + b;
        let g = T::grad(&[z], &[b]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[b],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_min() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_min(v, &[1], false); // keep_dims=false
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_min_keep() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_min(v, &[1], true); // keep_dims=true
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_max() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_max(v, &[1], false); // keep_dims=false
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_max_keep() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_max(v, &[1], true); // keep_dims=true
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_mean() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_mean(v, &[1], false); // keep_dims=false
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_mean_keep() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_mean(v, &[1], true); // keep_dims=true
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_sum() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_sum(v, &[1], false); // keep_dims=false
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_sum_keep() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_sum(v, &[1], true); // keep_dims=true
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reduce_prod() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_prod(v, &[1], false); // keep_dims=false
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn maximum() {
    let mut env = ag::VariableEnvironment::new();
    let v1 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v2 = env.slot().set(ndarray::arr1(&[4., 5., 6.]));
    env.run(|graph| {
        let v1 = graph.variable(v1);
        let v2 = graph.variable(v2);
        let z = T::maximum(v1, v2);
        let g = T::grad(&[z], &[v1, v2]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v1, v2],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn minimum() {
    let mut env = ag::VariableEnvironment::new();
    let v1 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    let v2 = env.slot().set(ndarray::arr1(&[4., 5., 6.]));
    env.run(|graph| {
        let v1 = graph.variable(v1);
        let v2 = graph.variable(v2);
        let z = T::minimum(v1, v2);
        let g = T::grad(&[z], &[v1, v2]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v1, v2],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn abs() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::abs(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn neg() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::neg(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn square() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::square(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reciprocal() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[2, 3], 1., 1.01));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::inv(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn lgamma() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[2, 3], 1., 1.01));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::lgamma_f64(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn dropout() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.random_uniform(&[2, 3], 1., 1.01));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::dropout(v, 0.01, true);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn transpose() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 2, 3, 4]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::transpose(v, &[2, 3, 0, 1]);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reshape_after_transpose() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3, 4]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::transpose(v, &[2, 1, 0]);
        let z = T::reshape(z, &[4, 6]);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn transpose_then_reshape_then_mm() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 2, 3, 4, 5]));
    let v2 = env.slot().set(rng.standard_normal(&[8, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let z = T::transpose(v, &[4, 2, 3, 0, 1]);
        let z = T::reshape(z, &[15, 8]);
        let z = T::matmul(z, v2);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn add() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = env.slot().set(rng.standard_normal(&[2, 2]));
    let b = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let a = graph.variable(a);
        let b = graph.variable(b);
        let z = a + b;
        let g = T::grad(&[z], &[a, b]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[a],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn mul() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = env.slot().set(rng.standard_normal(&[2, 2]));
    let b = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let a = graph.variable(a);
        let b = graph.variable(b);
        let z = a * b;
        let g = T::grad(&[z], &[a, b]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[a],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn sigmoid() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::sigmoid(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn elu() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::elu(v, 1.);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn relu() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ag::ndarray::arr1(&[0.2, 0.5]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::relu(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn softplus() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::softplus(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn logsumexp() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reduce_logsumexp(v, 1, true);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn log_softmax() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::log_softmax(v, 1);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn softmax_cross_entropy() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let t = T::convert_to_tensor(ndarray::arr2(&[[1., 0., 0.]]), graph);
        let v = graph.variable(v);
        let z = T::softmax_cross_entropy(v, t);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn sigmoid_cross_entropy() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[1, 3]));
    env.run(|graph| {
        let t = T::convert_to_tensor(rng.standard_normal(&[1, 3]), graph);
        let v = graph.variable(v);
        let z = T::sigmoid_cross_entropy(v, t);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn sparse_softmax_cross_entropy() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[2, 3]));
    env.run(|graph| {
        let t = T::convert_to_tensor(ndarray::arr1(&[1., 0.]), graph);
        let v = graph.variable(v);
        let z = T::sparse_softmax_cross_entropy(v, t);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn gather() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[5, 4, 8, 2]));
    env.run(|graph| {
        let v = graph.variable(v);
        let x = T::convert_to_tensor(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]), graph);
        let z = T::gather(v, x, 2);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn concat() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v1 = env.slot().set(rng.standard_normal(&[1, 2]));
    let v2 = env.slot().set(rng.standard_normal(&[1, 2]));
    env.run(|graph| {
        let v1 = graph.variable(v1);
        let v2 = graph.variable(v2);
        let z = T::concat(&[v1, v2], 1);
        let g = T::grad(&[z], &[v1]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v1, v2],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn slice() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::slice(v, &[0, 0], &[-1, 2]); // numpy equivalent is v[:, 0:2]
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn split() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[3, 7, 5]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::split(v, &[2, 3, 2], 1);
        let g = T::grad(&[&z[1]], &[v]);
        ag::test_helper::check_theoretical_grads(
            z[1],
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn flatten() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::flatten(v);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn reshape() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reshape(v, &[4, 2, 2]);
        let g = T::grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(
            z,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
// zero grad
fn reshape_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let v = env.slot().set(rng.standard_normal(&[4, 4]));
    env.run(|graph| {
        let v = graph.variable(v);
        let z = T::reshape(&v, &[4, 2, 2]);
        let g = T::grad(&[z], &[v])[0];
        let gg = T::grad(&[g], &[v]);
        ag::test_helper::check_theoretical_grads(
            g,
            gg.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn conv2d_transpose() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[3, 2, 2, 2]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable(x);
        let w = graph.variable(w);
        let y = T::conv2d_transpose(x, w, 0, 1);
        let g = T::grad(&[y], &[w]);
        ag::test_helper::check_theoretical_grads(y, &g, &[w], ag::Feeder::new(), 1e-3, 1e-2, graph);
    });
}

#[test]
// zero grad
fn conv2d_transpose_filter_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 2, 2, 2]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable(x);
        let w = graph.variable(w);
        let y = T::conv2d_transpose(x, w, 0, 1);
        let g = T::grad(&[y], &[w])[0];
        let gg = T::grad(&[g], &[w]);
        ag::test_helper::check_theoretical_grads(
            g,
            &gg,
            &[w],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
// zero grad
fn conv2d_filter_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable(x);
        let w = graph.variable(w);
        let y = T::conv2d(x, w, 0, 1);
        let g = T::grad(&[y], &[w])[0];
        let gg = T::grad(&[g], &[w]);
        ag::test_helper::check_theoretical_grads(
            g,
            &gg,
            &[w],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
fn conv2d_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    let gy = env.slot().set(ag::ndarray_ext::ones(&[2, 2, 2, 2]));
    env.run(|graph| {
        let x = graph.variable(x);
        let w = graph.variable(w);
        let y = T::conv2d(x, w, 0, 1);
        let gy = graph.variable(gy);
        unsafe {
            let g = T::grad_with_default(&[y], &[x], &[gy])[0];
            let gg = T::grad(&[g], &[gy])[0];
            ag::test_helper::check_theoretical_grads(
                g,
                &[gg],
                &[gy],
                ag::Feeder::new(),
                1e-3,
                1e-2,
                graph,
            );
        }
    });
}

#[test]
fn conv2d_xw_grad() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 2, 2]));
    env.run(|graph| {
        let x = graph.variable(x);
        let w = graph.variable(w);
        let y = T::conv2d(x, w, 0, 1);
        let g = T::grad(&[y], &[w])[0];
        let gg = T::grad(&[g], &[x]);
        ag::test_helper::check_theoretical_grads(
            g,
            &gg,
            &[x],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
fn conv2d() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let x = env.slot().set(rng.standard_normal(&[2, 3, 5, 5]));
    let w = env.slot().set(rng.standard_normal(&[2, 3, 3, 3]));
    env.run(|graph| {
        let x = graph.variable(x);
        let w = graph.variable(w);
        let y = T::conv2d(x, w, 1, 2);
        let g = T::grad(&[y], &[x, w]);
        ag::test_helper::check_theoretical_grads(
            y,
            &g,
            &[x, w],
            ag::Feeder::new(),
            1e-3,
            1e-2,
            graph,
        );
    });
}

#[test]
fn max_pool2d() {
    let mut env = ag::VariableEnvironment::new();
    let x = env.slot().set(ndarray::Array::linspace(0., 1., 9));
    env.run(|graph| {
        let x = graph.variable(x);
        let y = T::max_pool2d(T::reshape(x, &[1, 1, 3, 3]), 2, 0, 1);
        let g = T::grad(&[y], &[x]);
        ag::test_helper::check_theoretical_grads(y, &g, &[x], ag::Feeder::new(), 1e-3, 1e-2, graph);
    });
}

#[test]
fn max_pool2d_grad() {
    let mut env = ag::VariableEnvironment::new();
    let x = env.slot().set(ndarray::Array::linspace(0., 1., 36));
    let gy = env.slot().set(
        ndarray::Array::linspace(0., 1., 16)
            .into_shape(ndarray::IxDyn(&[2, 2, 2, 2]))
            .unwrap(),
    );
    env.run(|graph| {
        let x = graph.variable(x);
        let y = T::max_pool2d(T::reshape(x, &[2, 2, 3, 3]), 2, 0, 1);
        let gy = graph.variable(gy);
        unsafe {
            let g = T::grad_with_default(&[y], &[x], &[gy])[0];
            let gg = T::grad(&[g], &[gy])[0];
            ag::test_helper::check_theoretical_grads(
                g,
                &[gg],
                &[gy],
                ag::Feeder::new(),
                1e-3,
                1e-2,
                graph,
            );
        }
    });
}

#[test]
fn tensordot() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = env.slot().set(rng.standard_normal(&[3, 4, 5]));
    env.run(|graph| {
        let a = graph.variable(a);
        let b = T::convert_to_tensor(rng.standard_normal(&[4, 3, 2]), graph);
        let c = T::tensordot(a, b, &[1, 0], &[0, 1]);
        let g = T::grad(&[c], &[a]);
        ag::test_helper::check_theoretical_grads(c, &g, &[a], ag::Feeder::new(), 1e-3, 1e-2, graph);
    });
}

#[test]
fn primitive_back_propagation_through_time() {
    let mut env = ag::VariableEnvironment::new();
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let lookup_table = env.slot().set(rng.standard_normal(&[5, 3]));
    // (vector_dim -> vocab)
    let wo = env.slot().set(rng.standard_normal(&[3, 5]));
    // (vector_dim -> vector_dim)
    let wh = env.slot().set(rng.standard_normal(&[3, 3]));

    env.run(|graph| {
        let max_sent = 3;
        let batch_size = 2;

        let lookup_table = graph.variable(lookup_table);
        // (vector_dim -> vocab)
        let wo = graph.variable(wo);
        // (vector_dim -> vector_dim)
        let wh = graph.variable(wh);

        // -- build graph for BPTT --
        let mut loss_buf = vec![];
        let mut h_buf = vec![graph.placeholder("", &[-1, max_sent])];
        let sentences = graph.placeholder("sents", &[-1, max_sent]);

        for i in 0..max_sent {
            // pick new word id
            let id = T::squeeze(T::slice(sentences, &[0, i], &[-1, i + 1]), &[-1]);

            let new_h = {
                // recall last h
                let last_h = h_buf.last().unwrap();
                // compute and accumulate `loss`
                loss_buf.push(T::sparse_softmax_cross_entropy(&T::matmul(last_h, wo), &id));
                // new `h`
                T::tanh(&(T::gather(&lookup_table, &id, 0) + T::matmul(last_h, wh)))
            };

            h_buf.push(new_h);
        }
        // last loss (after processed whole sequence)
        let loss = *loss_buf.last().unwrap();

        // inputs (batch_size=2, sentence_len=4)
        let params = &[lookup_table, wo, wh];
        let g = T::grad(&[loss], params);
        let mut feeder = ag::Feeder::new();
        let sents = ndarray::arr2(&[[2., 3., 1.], [0., 2., 0.]]).into_dyn();
        let h_buf_default = rng.standard_normal(&[batch_size, 3]);
        feeder.push(sentences, sents.view());
        feeder.push(h_buf[0], h_buf_default.view());
        ag::test_helper::check_theoretical_grads(
            loss,
            g.as_slice(),
            params,
            feeder,
            1e-3,
            1e-3,
            graph,
        );
    });
}
