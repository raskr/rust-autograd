extern crate autograd as ag;
extern crate ndarray;
use ag::tensor::Constant;
use ag::tensor::Variable;

use ag::with;

#[test]
fn get() {
    with(|graph| {
        let v = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = 2. * v;
        let z = a.access_elem(1);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn add_n() {
    with(|graph| {
        let v1 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let v3 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let z = graph.add_n(&[v1, v2, v3]);
        let g = graph.grad(&[z], &[v2]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v2], &[], 1e-3, 1e-3);
    });
}

#[test]
fn clip() {
    with(|graph| {
        let v = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let z = graph.clip(v, 1.5, 2.5);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn asinh() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.asinh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn acosh() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.acosh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn atanh() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.atanh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn sinh() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.sinh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn cosh() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.cosh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn tanh() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.tanh(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn asin() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.asin(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
    });
}

#[test]
fn acos() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.acos(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn atan() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.atan(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn sin() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.sin(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn cos() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.cos(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn tan() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
        let z = graph.tan(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
    });
}

#[test]
fn pow() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1));
        let z = graph.pow(v, 1.1);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn exp() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1));
        let z = graph.exp(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
    });
}

#[test]
fn log() {
    with(|graph| {
        use std::f64;
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 1., 1.1));
        let z = graph.log(v, f64::consts::E);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
    });
}

#[test]
fn expand_dims() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3]));
        let z = graph.expand_dims(v, &[0, 2]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn squeeze() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 1, 2, 1]));
        let z = graph.squeeze(v, &[3, 1]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn g_matmul() {
    with(|graph| {
        let a = graph.constant(ag::ndarray_ext::standard_normal::<f64>(&[4, 2]));
        let v = graph.variable(ag::ndarray_ext::standard_normal::<f64>(&[2, 3]));
        let z = graph.matmul(a, v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 0.005);
    });
}

#[test]
fn batch_matmul() {
    with(|graph| {
        let a = graph.constant(ag::ndarray_ext::standard_normal(&[2, 4, 2]));
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2, 3]));
        let z = graph.batch_matmul(a, v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn implicit_broadcast() {
    with(|graph| {
        let x = graph.constant(ag::ndarray_ext::standard_normal(&[4, 3]));
        let b = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
        let z = x + b;
        let g = graph.grad(&[z], &[b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[b], &[], 1e-3, 1e-3);
    });
}

#[test]
fn wx_plus_b() {
    with(|graph| {
        let x = graph.constant(ag::ndarray_ext::standard_normal(&[4, 2]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
        let b = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
        let z = graph.matmul(x, w) + b;
        let g = graph.grad(&[z], &[b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[b], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_min() {
    with(|graph| {
        let v = graph.variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
        let z = graph.reduce_min(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_min_keep() {
    with(|graph| {
        let v = graph.variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
        let z = graph.reduce_min(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_max() {
    with(|graph| {
        let v = graph.variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
        let z = graph.reduce_max(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_max_keep() {
    with(|graph| {
        let v = graph.variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
        let z = graph.reduce_max(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_mean() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
        let z = graph.reduce_mean(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_mean_keep() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
        let z = graph.reduce_mean(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_sum() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
        let z = graph.reduce_sum(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_sum_keep() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
        let z = graph.reduce_sum(v, &[1], true); // keep_dims=true
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reduce_prod() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
        let z = graph.reduce_prod(v, &[1], false); // keep_dims=false
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn maximum() {
    with(|graph| {
        let v1 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let v2 = graph.variable(ndarray::arr1(&[4., 5., 6.]));
        let z = graph.maximum(v1, v2);
        let g = graph.grad(&[z], &[v1, v2]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v1, v2], &[], 1e-3, 1e-3);
    });
}

#[test]
fn minimum() {
    with(|graph| {
        let v1 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let v2 = graph.variable(ndarray::arr1(&[4., 5., 6.]));
        let z = graph.minimum(v1, v2);
        let g = graph.grad(&[z], &[v1, v2]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v1, v2], &[], 1e-3, 1e-3);
    });
}

#[test]
fn abs() {
    with(|graph| {
        let v = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let z = graph.abs(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn neg() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
        let z = graph.neg(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn square() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
        let z = graph.square(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reciprocal() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::random_uniform(&[2, 3], 1., 1.01));
        let z = graph.reciprocal(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn transpose() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 2, 3, 4]));
        let z = graph.transpose(v, &[2, 3, 0, 1]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reshape_after_transpose() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 4]));
        let z = graph.transpose(v, &[2, 1, 0]);
        let z = graph.reshape(z, &[4, 6]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn transpose_then_reshape_then_mm() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 2, 3, 4, 5]));
        let v2 = graph.variable(ag::ndarray_ext::standard_normal(&[8, 2]));
        let z = graph.transpose(v, &[4, 2, 3, 0, 1]);
        let z = graph.reshape(z, &[15, 8]);
        let z = graph.matmul(z, v2);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn add() {
    with(|graph| {
        let a = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
        let b = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
        let z = a + b;
        let g = graph.grad(&[z], &[a, b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[a], &[], 1e-3, 1e-3);
    });
}

#[test]
fn mul() {
    with(|graph| {
        let a = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
        let b = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
        let z = a * b;
        let g = graph.grad(&[z], &[a, b]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[a], &[], 1e-3, 1e-3);
    });
}

#[test]
fn sigmoid() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
        let z = graph.sigmoid(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn elu() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
        let z = graph.elu(v, 1.);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn relu() {
    with(|graph| {
        let v = graph.variable(ag::ndarray::arr1(&[0.2, 0.5]));
        let z = graph.relu(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn softplus() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
        let z = graph.softplus(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn logsumexp() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
        let z = graph.reduce_logsumexp(v, 1, true);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn log_softmax() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
        let z = graph.log_softmax(v, 1);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn softmax_cross_entropy() {
    with(|graph| {
        let t = graph.constant(ndarray::arr2(&[[1., 0., 0.]]));
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
        let z = graph.softmax_cross_entropy(v, t);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn sigmoid_cross_entropy() {
    with(|graph| {
        let t = graph.constant(ag::ndarray_ext::standard_normal(&[1, 3]));
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
        let z = graph.sigmoid_cross_entropy(v, t);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn sparse_softmax_cross_entropy() {
    with(|graph| {
        let t = graph.constant(ndarray::arr1(&[1., 0.]));
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
        let z = graph.sparse_softmax_cross_entropy(v, t);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn gather() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[5, 4, 8, 2]));
        let x = graph.constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));
        let z = graph.gather(v, x, 2);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn concat() {
    with(|graph| {
        let v1 = graph.variable(ag::ndarray_ext::standard_normal(&[1, 2]));
        let v2 = graph.variable(ag::ndarray_ext::standard_normal(&[1, 2]));
        let z = graph.concat(&[v1, v2], 1);
        let g = graph.grad(&[z], &[v1]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v1, v2], &[], 1e-3, 1e-3);
    });
}

#[test]
fn slice() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
        let z = graph.slice(v, &[0, 0], &[-1, 2]); // numpy equivalent is v[:, 0:2]
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn split() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 7, 5]));
        let z = graph.split(v, &[2, 3, 2], 1);
        let g = graph.grad(&[&z[1]], &[v]);
        ag::test_helper::check_theoretical_grads(z[1], g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn flatten() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
        let z = graph.flatten(v);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn reshape() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
        let z = graph.reshape(v, &[4, 2, 2]);
        let g = graph.grad(&[z], &[v]);
        ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
#[should_panic]
fn reshape_grad() {
    with(|graph| {
        let v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
        let z = graph.reshape(&(v), &[4, 2, 2]);
        let g = graph.grad(&[z], &[v])[0];
        let gg = graph.grad(&[g], &[v]);
        ag::test_helper::check_theoretical_grads(g, gg.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn conv2d_transposea() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2, 2]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
        let y = graph.conv2d_transpose(x, w, 0, 1);
        let g = graph.grad(&[y], &[w]);
        ag::test_helper::check_theoretical_grads(y, &g, &[w], &[], 1e-3, 1e-2);
    });
}

#[test]
#[should_panic]
fn conv2d_transpose_filter_grad() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2, 2, 2]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
        let y = graph.conv2d_transpose(x, w, 0, 1);
        let g = graph.grad(&[y], &[w])[0];
        let gg = graph.grad(&[g], &[w]);
        ag::test_helper::check_theoretical_grads(g, &gg, &[w], &[], 1e-3, 1e-2);
    });
}

#[test]
#[should_panic]
fn conv2d_filter_grad() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
        let y = graph.conv2d(x, w, 0, 1);
        let g = graph.grad(&[y], &[w])[0];
        let gg = graph.grad(&[g], &[w]);
        ag::test_helper::check_theoretical_grads(g, &gg, &[w], &[], 1e-3, 1e-2);
    });
}

#[test]
fn conv2d_grad() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
        let y = graph.conv2d(x, w, 0, 1);
        let gy = graph.variable(ag::ndarray_ext::ones(&[2, 2, 2, 2]));
        unsafe {
            let g = graph.grad_with_default(&[y], &[x], &[gy])[0];
            let gg = graph.grad(&[g], &[gy])[0];
            ag::test_helper::check_theoretical_grads(g, &[gg], &[gy], &[], 1e-3, 1e-2);
        }
    });
}

#[test]
fn conv2d_xw_grad() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
        let y = graph.conv2d(x, w, 0, 1);
        let g = graph.grad(&[y], &[w])[0];
        let gg = graph.grad(&[g], &[x]);
        ag::test_helper::check_theoretical_grads(g, &gg, &[x], &[], 1e-3, 1e-2);
    });
}

#[test]
#[should_panic]
fn conv2d_x_grad() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
        let y = graph.conv2d(x, w, 0, 1);
        let g = graph.grad(&[y], &[x])[0];
        let gg = graph.grad(&[g], &[x]); // can't differentiate with x twice
        ag::test_helper::check_theoretical_grads(y, &gg, &[x], &[], 1e-3, 1e-2);
    });
}

#[test]
fn conv2d() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
        let w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3, 3, 3]));
        let y = graph.conv2d(x, w, 1, 2);
        let g = graph.grad(&[y], &[x, w]);
        ag::test_helper::check_theoretical_grads(y, &g, &[x, w], &[], 1e-3, 1e-2);
    });
}

#[test]
fn max_pool2d() {
    with(|graph| {
        let x = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2, 3, 3]));
        let y = graph.max_pool2d(x, 2, 0, 1);
        let g = graph.grad(&[y], &[x]);
        ag::test_helper::check_theoretical_grads(y, &g, &[x], &[], 1e-3, 1e-2);
    });
}

#[test]
fn max_pool2d_grad() {
    with(|graph| {
        let arr_x = ag::ndarray_ext::standard_normal(&[2, 2, 3, 3]);
        let arr_gx = ag::ndarray_ext::standard_normal(&[2, 2, 2, 2]);
        let x = graph.variable(arr_x);
        let y = graph.max_pool2d(x, 2, 0, 1);
        let gy = graph.variable(arr_gx);
        unsafe {
            let g = graph.grad_with_default(&[y], &[x], &[gy])[0];
            let gg = graph.grad(&[g], &[gy])[0];
            ag::test_helper::check_theoretical_grads(g, &[gg], &[gy], &[], 1e-3, 1e-2);
        }
    });
}

#[test]
fn tensordot() {
    with(|graph| {
        let a = graph.variable(ag::ndarray_ext::standard_normal(&[3, 4, 5]));
        let b = graph.constant(ag::ndarray_ext::standard_normal(&[4, 3, 2]));
        let c = graph.tensordot(a, b, &[1, 0], &[0, 1]);
        let g = graph.grad(&[c], &[a]);
        ag::test_helper::check_theoretical_grads(c, &g, &[a], &[], 1e-3, 1e-2);
    });
}

#[test]
fn primitive_back_propagation_through_time() {
    with(|graph| {
        let max_sent = 3;
        let batch_size = 2;

        let lookup_table = graph.variable(ag::ndarray_ext::standard_normal(&[5, 3]));
        // (vector_dim -> vocab)
        let wo = graph.variable(ag::ndarray_ext::standard_normal(&[3, 5]));
        // (vector_dim -> vector_dim)
        let wh = graph.variable(ag::ndarray_ext::standard_normal(&[3, 3]));

        // -- build graph for BPTT --
        let mut loss_buf = vec![];
        let mut h_buf = vec![graph.placeholder(&[-1, max_sent])];
        let sentences = graph.placeholder(&[-1, max_sent]);

        for i in 0..max_sent {
            // pick new word id
            let id = graph.squeeze(graph.slice(sentences, &[0, i], &[-1, i + 1]), &[-1]);

            let new_h = {
                // recall last h
                let last_h = h_buf.last().unwrap();
                // compute and accumulate `loss`
                loss_buf.push(graph.sparse_softmax_cross_entropy(&graph.matmul(last_h, wo), &id));
                // new `h`
                graph.tanh(&(graph.gather(&lookup_table, &id, 0) + graph.matmul(last_h, wh)))
            };

            h_buf.push(new_h);
        }
        // last loss (after processed whole sequence)
        let loss = *loss_buf.last().unwrap();

        // inputs (batch_size=2, sentence_len=4)
        let params = &[lookup_table, wo, wh];
        let g = graph.grad(&[loss], params);
        ag::test_helper::check_theoretical_grads(
            loss,
            g.as_slice(),
            params,
            &[
                sentences.given(
                    ndarray::arr2(&[[2., 3., 1.], [0., 2., 0.]])
                        .into_dyn()
                        .view(),
                ),
                h_buf[0].given(ag::ndarray_ext::standard_normal(&[batch_size, 3]).view()),
            ],
            1e-3,
            1e-3,
        );
    });
}
