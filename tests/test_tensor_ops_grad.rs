extern crate autograd as ag;
extern crate ndarray;


#[test]
fn get()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
    let ref a: ag::Tensor = 2 * v;
    let ref z = a.get(1);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn add_n()
{
    let mut ctx = ag::Context::new();
    let ref v1 = ag::variable(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
    let ref v2 = ag::variable(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
    let ref v3 = ag::variable(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
    let ref z = ag::add_n(&[v1, v2, v3]);
    let ref g = ag::grad_with_default(&[z], &[v2], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v2], ctx, 1e-3, 1e-3);
}

#[test]
fn clip()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
    let ref z = ag::clip(v, 1.5, 2.5);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn asinh()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::asinh(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn acosh()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::acosh(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn atanh()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::atanh(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn sinh()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::sinh(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn cosh()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::cosh(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn tanh()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::tanh(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn asin()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::asin(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-2);
}

#[test]
fn acos()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::acos(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn atan()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::atan(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn sin()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::sin(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn cos()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::cos(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn tan()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2), &mut ctx);
    let ref z = ag::tan(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-2);
}

#[test]
fn pow()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1), &mut ctx);
    let ref z = ag::pow(v, 1.1);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn exp()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1), &mut ctx);
    let ref z = ag::exp(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-2);
}

#[test]
fn log()
{
    use std::f32;
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 1., 1.1), &mut ctx);
    let ref z = ag::log(v, f32::consts::E);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-2);
}

#[test]
fn expand_dims()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3]), &mut ctx);
    let ref z = ag::expand_dims(v, &[0, 2]);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn squeeze()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 1, 2, 1]), &mut ctx);
    let ref z = ag::squeeze(v, &[3, 1]);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn matmul()
{
    let mut ctx = ag::Context::new();
    let ref a = ag::constant(ag::ndarray_ext::standard_normal(&[4, 2]), &mut ctx);
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]), &mut ctx);
    let ref z = ag::matmul(a, v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn batch_matmul()
{
    let mut ctx = ag::Context::new();
    let ref a = ag::constant(ag::ndarray_ext::standard_normal(&[2, 4, 2]), &mut ctx);
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2, 3]), &mut ctx);
    let ref z = ag::batch_matmul(a, v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn implicit_broadcast()
{
    let mut ctx = ag::Context::new();
    let ref x = ag::constant(ag::ndarray_ext::standard_normal(&[4, 3]), &mut ctx);
    let ref b = ag::variable(ag::ndarray_ext::zeros(&[1, 3]), &mut ctx);
    let ref z = x + b;
    let ref g = ag::grad_with_default(&[z], &[b], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[b], ctx, 1e-3, 1e-3);
}

#[test]
fn wx_plus_b()
{
    let mut ctx = ag::Context::new();
    let ref x = ag::constant(ag::ndarray_ext::standard_normal(&[4, 2]), &mut ctx);
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]), &mut ctx);
    let ref b = ag::variable(ag::ndarray_ext::zeros(&[1, 3]), &mut ctx);
    let ref z = ag::matmul(x, w) + b;
    let ref g = ag::grad_with_default(&[z], &[b], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[b], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_min()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_min(v, &[1], false); // keep_dims=false
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_min_keep()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_min(v, &[1], true); // keep_dims=true
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_max()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_max(v, &[1], false); // keep_dims=false
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_max_keep()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_max(v, &[1], true); // keep_dims=true
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_mean()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_mean(v, &[1], false); // keep_dims=false
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_mean_keep()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_mean(v, &[1], true); // keep_dims=true
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_sum()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_sum(v, &[1], false); // keep_dims=false
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_sum_keep()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_sum(v, &[1], true); // keep_dims=true
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reduce_prod()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]), &mut ctx);
    let ref z = ag::reduce_prod(v, &[1], false); // keep_dims=false
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn abs()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_uniform(&[2, 3]), &mut ctx);
    let ref z = ag::abs(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn neg()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_uniform(&[2, 3]), &mut ctx);
    let ref z = ag::neg(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn square()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_uniform(&[2, 3]), &mut ctx);
    let ref z = ag::square(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reciprocal()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[2, 3], 1., 1.01), &mut ctx);
    let ref z = ag::reciprocal(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn transpose()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::zeros(&[1, 2, 3, 4]), &mut ctx);
    let ref z = ag::transpose(v, &[2, 3, 0, 1]);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&[3, 4, 1, 2])]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reshape_after_transpose()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::zeros(&[1, 2, 3, 4, 5]), &mut ctx);
    let ref z = ag::transpose(v, &[4, 2, 3, 0, 1]);
    let ref z = ag::reshape(z, &[15, 8]);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn add_inplace()
{
    let mut ctx = ag::Context::new();
    let a = ag::zeros(&[2, 2]) + ag::ones(&[2, 2]);
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref c = ag::add_inplace(a, b);
    let ref g = ag::grad_with_default(&[c], &[b], &[&ag::ones(&c.shape())]);
    ag::test_helper::gradient_check(c, g.as_slice(), &[b], ctx, 1e-3, 1e-3);
}

#[test]
fn sub_inplace()
{
    let mut ctx = ag::Context::new();
    let a = ag::zeros(&[2, 2]) + ag::ones(&[2, 2]);
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref c = ag::sub_inplace(a, b);
    let ref g = ag::grad_with_default(&[c], &[b], &[&ag::ones(&c.shape())]);
    ag::test_helper::gradient_check(c, g.as_slice(), &[b], ctx, 1e-3, 1e-3);
}

#[test]
fn add()
{
    let mut ctx = ag::Context::new();
    let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref z = a + b;
    let ref g = ag::grad_with_default(&[z], &[a, b], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[a], ctx, 1e-3, 1e-3);
}

#[test]
fn mul()
{
    let mut ctx = ag::Context::new();
    let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref z = a * b;
    let ref g = ag::grad_with_default(&[z], &[a, b], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[a], ctx, 1e-3, 1e-3);
}

#[test]
fn sigmoid()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref z = ag::sigmoid(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn elu()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref z = ag::elu(v, 1.);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn relu()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref z = ag::relu(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn softplus()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]), &mut ctx);
    let ref z = ag::softplus(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn logsumexp()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]), &mut ctx);
    let ref z = ag::reduce_logsumexp(v, 1, true);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&[1, 3])]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn log_softmax()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]), &mut ctx);
    let ref z = ag::log_softmax(v, 1);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn softmax_cross_entropy()
{
    let mut ctx = ag::Context::new();
    let ref t = ag::constant(ndarray::arr2(&[[1., 0., 0.]]), &mut ctx);
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]), &mut ctx);
    let ref z = ag::softmax_cross_entropy(v, t);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn sigmoid_cross_entropy()
{
    let mut ctx = ag::Context::new();
    let ref t = ag::constant(ag::ndarray_ext::standard_normal(&[1, 3]), &mut ctx);
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]), &mut ctx);
    let ref z = ag::sigmoid_cross_entropy(v, t);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn sparse_softmax_cross_entropy()
{
    let mut ctx = ag::Context::new();
    let ref t = ag::constant(ndarray::arr1(&[1., 0.]), &mut ctx);
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]), &mut ctx);
    let ref z = ag::sparse_softmax_cross_entropy(v, t);
    let ref g = ag::grad(&[z], &[v]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn gather()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::zeros(&[5, 4, 8, 2]), &mut ctx);
    let ref x = ag::constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]), &mut ctx);
    let ref z = ag::gather(v, x, 2);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&[5, 4, 2, 3, 2])]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn concat()
{
    let mut ctx = ag::Context::new();
    let ref v1 = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2]), &mut ctx);
    let ref v2 = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2]), &mut ctx);
    let ref z = ag::concat(&[v1, v2], 1);
    let ref g = ag::grad_with_default(&[z], &[v1], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v1], ctx, 1e-3, 1e-3);
    // FIXME: uncommenting below causes SEGV
    // ag::helper::gradient_check(z, &[v1, v2], g.as_slice(), &ag::Input::new(), 1e-3, 1e-3);
}

#[test]
fn slice()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]), &mut ctx);
    let ref z = ag::slice(v, &[0, 0], &[-1, 2]); // numpy equivalent is v[:, 0:2]
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn split()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 7, 5]), &mut ctx);
    let ref z = ag::split(v, &[2, 3, 2], 1);
    let ref g = ag::grad_with_default(&[&z[1]], &[v], &[&ag::ones(&z[1].shape())]);
    ag::test_helper::gradient_check(&z[1], g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn flatten()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]), &mut ctx);
    let ref z = ag::flatten(v);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reshape()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]), &mut ctx);
    let ref z = ag::reshape(v, &[4, 2, 2]);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn reshape_grad()
{
    let mut ctx = ag::Context::new();
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]), &mut ctx);
    let ref z = ag::reshape(&(v * v), &[4, 2, 2]);
    let ref g = ag::grad_with_default(&[z], &[v], &[&ag::ones(&z.shape())])[0];
    let ref gg = ag::grad_with_default(&[g], &[v], &[&ag::ones(&g.shape())]);
    ag::test_helper::gradient_check(g, gg.as_slice(), &[v], ctx, 1e-3, 1e-3);
}

#[test]
fn primitive_back_propagation_through_time()
{
    let max_sent = 3;
    let batch_size = 2;

    let mut ctx = ag::Context::new();
    let ref lookup_table = ag::variable(ag::ndarray_ext::standard_normal(&[5, 3]), &mut ctx);
    // (vector_dim -> vocab)
    let ref wo = ag::variable(ag::ndarray_ext::standard_normal(&[3, 5]), &mut ctx);
    // (vector_dim -> vector_dim)
    let ref wh = ag::variable(ag::ndarray_ext::standard_normal(&[3, 3]), &mut ctx);

    // -- build graph for BPTT --
    let mut loss_buf = vec![];
    let mut h_buf = vec![ag::placeholder(&[-1, max_sent])];
    let sentences = ag::placeholder(&[-1, max_sent]);

    for i in 0..max_sent {
        // pick new word id
        let id = ag::slice(&sentences, &[0, i], &[-1, i + 1]);

        let new_h = {
            // recall last h
            let last_h = h_buf.last().unwrap();
            // compute and accumulate `loss`
            loss_buf.push(ag::sparse_softmax_cross_entropy(
                &ag::matmul(last_h, wo),
                &id,
            ));
            // new `h`
            ag::tanh(
                &(ag::gather(&lookup_table, &id, 0) + ag::matmul(last_h, wh)),
            )
        };

        h_buf.push(new_h);
    }
    // last loss (after processed whole sequence)
    let loss = loss_buf.last().unwrap();

    // inputs (batch_size=2, sentence_len=4)
    ctx.feed_input(
        &sentences,
        ndarray::arr2(&[[2., 3., 1., 3.], [0., 2., 0., 1.]]),
    );
    ctx.feed_input(&h_buf[0], ag::ndarray_ext::zeros(&[batch_size, 3]));
    let params = &[lookup_table, wo, wh];
    let ref g = ag::grad_with_default(&[loss], params, &[&ag::ones(&loss.shape())]);
    ag::test_helper::gradient_check(loss, g.as_slice(), params, ctx, 1e-3, 1e-3);
}
