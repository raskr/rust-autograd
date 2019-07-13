extern crate autograd as ag;
extern crate ndarray;

#[test]
fn get() {
    let ref v = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref a: ag::Tensor<f64> = 2. * v;
    let ref z = a.get(1);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn add_n() {
    let ref v1 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v2 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v3 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref z = ag::add_n(&[v1, v2, v3]);
    let ref g = ag::grad(&[z], &[v2]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v2], &[], 1e-3, 1e-3);
}

#[test]
fn clip() {
    let ref v = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref z = ag::clip(v, 1.5, 2.5);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn asinh() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::asinh(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn acosh() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::acosh(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn atanh() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::atanh(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn sinh() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::sinh(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn cosh() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::cosh(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn tanh() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::tanh(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn asin() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::asin(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
}

#[test]
fn acos() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::acos(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn atan() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::atan(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn sin() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::sin(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn cos() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::cos(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn tan() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::tan(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
}

#[test]
fn pow() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1));
    let ref z = ag::pow(v, 1.1);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn exp() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1));
    let ref z = ag::exp(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
}

#[test]
fn log() {
    use std::f64;
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 1., 1.1));
    let ref z = ag::log(v, f64::consts::E);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-2);
}

#[test]
fn expand_dims() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3]));
    let ref z = ag::expand_dims(v, &[0, 2]);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn squeeze() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 1, 2, 1]));
    let ref z = ag::squeeze(v, &[3, 1]);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn matmul() {
    let ref a = ag::constant(ag::ndarray_ext::standard_normal::<f64>(&[4, 2]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal::<f64>(&[2, 3]));
    let ref z = ag::matmul(a, v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 0.005);
}

#[test]
fn batch_matmul() {
    let ref a = ag::constant(ag::ndarray_ext::standard_normal(&[2, 4, 2]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2, 3]));
    let ref z = ag::batch_matmul(a, v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn implicit_broadcast() {
    let ref x = ag::constant(ag::ndarray_ext::standard_normal(&[4, 3]));
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = x + b;
    let ref g = ag::grad(&[z], &[b]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[b], &[], 1e-3, 1e-3);
}

#[test]
fn wx_plus_b() {
    let ref x = ag::constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::matmul(x, w) + b;
    let ref g = ag::grad(&[z], &[b]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[b], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_min() {
    let ref v = ag::variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    let ref z = ag::reduce_min(v, &[1], false); // keep_dims=false
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_min_keep() {
    let ref v = ag::variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    let ref z = ag::reduce_min(v, &[1], true); // keep_dims=true
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_max() {
    let ref v = ag::variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    let ref z = ag::reduce_max(v, &[1], false); // keep_dims=false
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_max_keep() {
    let ref v = ag::variable(ndarray::arr2(&[[0., 1.], [3., 2.]]));
    let ref z = ag::reduce_max(v, &[1], true); // keep_dims=true
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_mean() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_mean(v, &[1], false); // keep_dims=false
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_mean_keep() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_mean(v, &[1], true); // keep_dims=true
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_sum() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_sum(v, &[1], false); // keep_dims=false
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_sum_keep() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_sum(v, &[1], true); // keep_dims=true
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reduce_prod() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_prod(v, &[1], false); // keep_dims=false
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn maximum() {
    let ref v1 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v2 = ag::variable(ndarray::arr1(&[4., 5., 6.]));
    let ref z = ag::maximum(v1, v2);
    let ref g = ag::grad(&[z], &[v1, v2]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v1, v2], &[], 1e-3, 1e-3);
}

#[test]
fn minimum() {
    let ref v1 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v2 = ag::variable(ndarray::arr1(&[4., 5., 6.]));
    let ref z = ag::minimum(v1, v2);
    let ref g = ag::grad(&[z], &[v1, v2]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v1, v2], &[], 1e-3, 1e-3);
}

#[test]
fn abs() {
    let ref v = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref z = ag::abs(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn neg() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::neg(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn square() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::square(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reciprocal() {
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[2, 3], 1., 1.01));
    let ref z = ag::reciprocal(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn transpose() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2, 3, 4]));
    let ref z = ag::transpose(v, &[2, 3, 0, 1]);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reshape_after_transpose() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 4]));
    let ref z = ag::transpose(v, &[2, 1, 0]);
    let ref z = ag::reshape(z, &[4, 6]);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn transpose_then_reshape_then_mm() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2, 3, 4, 5]));
    let ref v2 = ag::variable(ag::ndarray_ext::standard_normal(&[8, 2]));
    let ref z = ag::transpose(v, &[4, 2, 3, 0, 1]);
    let ref z = ag::reshape(z, &[15, 8]);
    let ref z = ag::matmul(z, v2);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn add() {
    let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = a + b;
    let ref g = ag::grad(&[z], &[a, b]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[a], &[], 1e-3, 1e-3);
}

#[test]
fn mul() {
    let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = a * b;
    let ref g = ag::grad(&[z], &[a, b]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[a], &[], 1e-3, 1e-3);
}

#[test]
fn sigmoid() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::sigmoid(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn elu() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::elu(v, 1.);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn relu() {
    let ref v = ag::variable(ag::ndarray::arr1(&[0.2, 0.5]));
    let ref z = ag::relu(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn softplus() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::softplus(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn logsumexp() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::reduce_logsumexp(v, 1, true);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn log_softmax() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::log_softmax(v, 1);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn softmax_cross_entropy() {
    let ref t = ag::constant(ndarray::arr2(&[[1., 0., 0.]]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::softmax_cross_entropy(v, t);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn sigmoid_cross_entropy() {
    let ref t = ag::constant(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::sigmoid_cross_entropy(v, t);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn sparse_softmax_cross_entropy() {
    let ref t = ag::constant(ndarray::arr1(&[1., 0.]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::sparse_softmax_cross_entropy(v, t);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn gather() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[5, 4, 8, 2]));
    let ref x = ag::constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));
    let ref z = ag::gather(v, x, 2);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn concat() {
    let ref v1 = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2]));
    let ref v2 = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2]));
    let ref z = ag::concat(&[v1, v2], 1);
    let ref g = ag::grad(&[z], &[v1]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v1, v2], &[], 1e-3, 1e-3);
}

#[test]
fn slice() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::slice(v, &[0, 0], &[-1, 2]); // numpy equivalent is v[:, 0:2]
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn split() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 7, 5]));
    let ref z = ag::split(v, &[2, 3, 2], 1);
    let ref g = ag::grad(&[&z[1]], &[v]);
    ag::test_helper::check_theoretical_grads(&z[1], g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn flatten() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::flatten(v);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn reshape() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::reshape(v, &[4, 2, 2]);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::check_theoretical_grads(z, g.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
#[should_panic]
fn reshape_grad() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::reshape(&(v), &[4, 2, 2]);
    let ref g = ag::grad(&[z], &[v])[0];
    let ref gg = ag::grad(&[g], &[v]);
    ag::test_helper::check_theoretical_grads(g, gg.as_slice(), &[v], &[], 1e-3, 1e-3);
}

#[test]
fn conv2d_transpose() {
    let ref x = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2, 2, 2]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
    let ref y = ag::conv2d_transpose(x, w, 0, 1);
    let ref g = ag::grad(&[y], &[w]);
    ag::test_helper::check_theoretical_grads(y, g, &[w], &[], 1e-3, 1e-2);
}

#[test]
#[should_panic]
fn conv2d_transpose_filter_grad() {
    let ref x = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2, 2, 2]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
    let ref y = ag::conv2d_transpose(x, w, 0, 1);
    let ref g = ag::grad(&[y], &[w])[0];
    let ref gg = ag::grad(&[g], &[w]);
    ag::test_helper::check_theoretical_grads(g, gg, &[w], &[], 1e-3, 1e-2);
}

#[test]
#[should_panic]
fn conv2d_filter_grad() {
    let ref x = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
    let ref y = ag::conv2d(x, w, 0, 1);
    let ref g = ag::grad(&[y], &[w])[0];
    let ref gg = ag::grad(&[g], &[w]);
    ag::test_helper::check_theoretical_grads(g, gg, &[w], &[], 1e-3, 1e-2);
}

#[test]
fn conv2d_grad() {
    let ref x = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
    let ref y = ag::conv2d(x, w, 0, 1);
    let ref gy = ag::variable(ag::ndarray_ext::ones(&[2, 2, 2, 2]));
    unsafe {
        let ref g = ag::grad_with_default(&[y], &[x], &[gy])[0];
        let ref gg = ag::grad(&[g], &[gy])[0];
        ag::test_helper::check_theoretical_grads(g, &[gg], &[gy], &[], 1e-3, 1e-2);
    }
}

#[test]
fn conv2d_xw_grad() {
    let ref x = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
    let ref y = ag::conv2d(x, w, 0, 1);
    let ref g = ag::grad(&[y], &[w])[0];
    let ref gg = ag::grad(&[g], &[x]);
    ag::test_helper::check_theoretical_grads(g, gg, &[x], &[], 1e-3, 1e-2);
}

#[test]
#[should_panic]
fn conv2d_x_grad() {
    let ref x = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 2, 2]));
    let ref y = ag::conv2d(x, w, 0, 1);
    let ref g = ag::grad(&[y], &[x])[0];
    let ref gg = ag::grad(&[g], &[x]); // can't differentiate with x twice
    ag::test_helper::check_theoretical_grads(y, gg, &[x], &[], 1e-3, 1e-2);
}

#[test]
fn conv2d() {
    let ref x = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 5, 5]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3, 3, 3]));
    let ref y = ag::conv2d(x, w, 1, 2);
    let ref g = ag::grad(&[y], &[x, w]);
    ag::test_helper::check_theoretical_grads(y, g, &[x, w], &[], 1e-3, 1e-2);
}

#[test]
fn max_pool2d() {
    let ref x = ag::variable(ag::ndarray_ext::range(&[2, 2, 3, 3]));
    let ref y = ag::max_pool2d(x, 2, 0, 1);
    let ref g = ag::grad(&[y], &[x]);
    ag::test_helper::check_theoretical_grads(y, g, &[x], &[], 1e-3, 1e-2);
}

#[test]
fn max_pool2d_grad() {
    let arr_x = ag::ndarray_ext::range(&[2, 2, 3, 3]);
    let arr_gx = ag::ndarray_ext::range(&[2, 2, 2, 2]);
    let ref x = ag::variable(arr_x);
    let ref y = ag::max_pool2d(x, 2, 0, 1);
    let ref gy = ag::variable(arr_gx);
    unsafe {
        let ref g = ag::grad_with_default(&[y], &[x], &[gy])[0];
        let ref gg = ag::grad(&[g], &[gy])[0];
        ag::test_helper::check_theoretical_grads(g, &[gg], &[gy], &[], 1e-3, 1e-2);
    }
}

#[test]
fn tensordot() {
    let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[3, 4, 5]));
    let ref b = ag::constant(ag::ndarray_ext::standard_normal(&[4, 3, 2]));
    let ref c = ag::tensordot(a, b, &[1, 0], &[0, 1]);
    let ref g = ag::grad(&[c], &[a]);
    ag::test_helper::check_theoretical_grads(c, g, &[a], &[], 1e-3, 1e-2);
}

#[test]
fn primitive_back_propagation_through_time() {
    let max_sent = 3;
    let batch_size = 2;

    let ref lookup_table = ag::variable(ag::ndarray_ext::standard_normal(&[5, 3]));
    // (vector_dim -> vocab)
    let ref wo = ag::variable(ag::ndarray_ext::standard_normal(&[3, 5]));
    // (vector_dim -> vector_dim)
    let ref wh = ag::variable(ag::ndarray_ext::standard_normal(&[3, 3]));

    // -- build graph for BPTT --
    let mut loss_buf = vec![];
    let mut h_buf = vec![ag::placeholder(&[-1, max_sent])];
    let ref sentences = ag::placeholder(&[-1, max_sent]);

    for i in 0..max_sent {
        // pick new word id
        let id = ag::squeeze(ag::slice(sentences, &[0, i], &[-1, i + 1]), &[-1]);

        let new_h = {
            // recall last h
            let last_h = h_buf.last().unwrap();
            // compute and accumulate `loss`
            loss_buf.push(ag::sparse_softmax_cross_entropy(
                &ag::matmul(last_h, wo),
                &id,
            ));
            // new `h`
            ag::tanh(&(ag::gather(&lookup_table, &id, 0) + ag::matmul(last_h, wh)))
        };

        h_buf.push(new_h);
    }
    // last loss (after processed whole sequence)
    let loss = loss_buf.last().unwrap();

    // inputs (batch_size=2, sentence_len=4)
    let params = &[lookup_table, wo, wh];
    let ref g = ag::grad(&[loss], params);
    ag::test_helper::check_theoretical_grads(
        loss,
        g.as_slice(),
        params,
        &[
            ag::Feed(
                sentences,
                ndarray::arr2(&[[2., 3., 1.], [0., 2., 0.]])
                    .into_dyn()
                    .view(),
            ),
            ag::Feed(
                &h_buf[0],
                ag::ndarray_ext::standard_normal(&[batch_size, 3]).view(),
            ),
        ],
        1e-3,
        1e-3,
    );
}
