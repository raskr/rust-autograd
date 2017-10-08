extern crate autograd as ag;
extern crate ndarray;


#[test]
fn get()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ndarray::arr1(&[1., 2., 3.]));
    let ref a: ag::Tensor = 2 * v;
    let ref z = a.get(1);
    let ref g = ag::gradients(&[z], &[v], &[None]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn add_n()
{
    let mut graph = ag::Graph::new();
    let ref v1 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v3 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
    let ref z = ag::add_n(&[v1, v2, v3]);
    let ref g = ag::gradients(&[z], &[v2], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v2], graph, 1e-3, 1e-3);
}

#[test]
fn clip()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ndarray::arr1(&[1., 2., 3.]));
    let ref z = ag::clip(v, 1.5, 2.5);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn asinh()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::asinh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn acosh()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::acosh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn atanh()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::atanh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn sinh()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::sinh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn cosh()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::cosh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn tanh()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::tanh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn asin()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::asin(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-2);
}

#[test]
fn acos()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::acos(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn atan()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::atan(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn sin()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::sin(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn cos()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::cos(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn tan()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::tan(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-2);
}

#[test]
fn pow()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::pow(v, 1.2);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn exp()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1));
    let ref z = ag::exp(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-2);
}

#[test]
fn log()
{
    use std::f32;
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::random_uniform(&[3], 1., 1.1));
    let ref z = ag::log(v, f32::consts::E);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-2);
}

#[test]
fn expand_dims()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3]));
    let ref z = ag::expand_dims(v, &[0, 2]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[1, 3, 1]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn squeeze()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 1, 2, 1]));
    let ref z = ag::squeeze(v, &[3, 1]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn matmul()
{
    let mut graph = ag::Graph::new();
    let ref a = graph.constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::matmul(a, v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[4, 3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn batch_matmul()
{
    let mut graph = ag::Graph::new();
    let ref a = graph.constant(ag::ndarray_ext::standard_normal(&[2, 4, 2]));
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2, 3]));
    let ref z = ag::batch_matmul(a, v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[2, 4, 3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn implicit_broadcast()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.constant(ag::ndarray_ext::standard_normal(&[4, 3]));
    let ref b = graph.variable(ag::ndarray_ext::zeros(&[1, 3]));
    let ref z = x + b;
    let ref g = ag::gradients(&[z], &[b], &[Some(&graph.ones(&[4, 3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[b], graph, 1e-3, 1e-3);
}

#[test]
fn xw_plus_b()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    let ref w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref b = graph.variable(ag::ndarray_ext::zeros(&[1, 3]));
    let ref z = ag::matmul(x, w) + b;
    let ref g = ag::gradients(&[z], &[b], &[Some(&graph.ones(&[4, 3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[b], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_min()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_min(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_min_keep()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_min(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_max()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_max(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_max_keep()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_max(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_mean()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_mean(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_mean_keep()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_mean(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_sum()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_sum(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_sum_keep()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_sum(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reduce_prod()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_prod(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reverse_axes()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.constant(ag::ndarray_ext::zeros(&[2, 3, 4, 5]));
    let ref z = ag::reverse_axes(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[5, 4, 3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn transpose()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.constant(ag::ndarray_ext::zeros(&[1, 2, 3, 4, 5]));
    let ref z = ag::transpose(v, &[4, 2, 3, 0, 1]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[5, 3, 4, 1, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reshape_after_transpose()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.constant(ag::ndarray_ext::zeros(&[1, 2, 3, 4, 5]));
    let ref z = ag::transpose(v, &[4, 2, 3, 0, 1]);
    let ref z = ag::reshape(z, &[15, 8]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[15, 8]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn add_inplace()
{
    let mut graph = ag::Graph::new();
    let a = graph.zeros(&[2, 2]) + graph.ones(&[2, 2]);
    let ref b = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref c = ag::add_inplace(a, b);
    let ref g = ag::gradients(&[c], &[b], &[Some(&graph.ones(&[2, 2]))]);
    ag::test_helper::gradient_check(c, g.as_slice(), &[b], graph, 1e-3, 1e-3);
}

#[test]
fn sub_inplace()
{
    let mut graph = ag::Graph::new();
    let a = graph.zeros(&[2, 2]) + graph.ones(&[2, 2]);
    let ref b = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref c = ag::sub_inplace(a, b);
    let ref g = ag::gradients(&[c], &[b], &[Some(&graph.ones(&[2, 2]))]);
    ag::test_helper::gradient_check(c, g.as_slice(), &[b], graph, 1e-3, 1e-3);
}

#[test]
fn add()
{
    let mut graph = ag::Graph::new();
    let ref a = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref b = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = a + b;
    let ref g = ag::gradients(&[z], &[a, b], &[Some(&graph.ones(&[2, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[a], graph, 1e-3, 1e-3);
}

#[test]
fn mul()
{
    let mut graph = ag::Graph::new();
    let ref a = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref b = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = a * b;
    let ref g = ag::gradients(&[z], &[a, b], &[Some(&graph.ones(&[2, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[a], graph, 1e-3, 1e-3);
}

#[test]
fn sigmoid()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::sigmoid(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[2, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn elu()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::elu(v, 1.);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[2, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn relu()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::relu(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[2, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn logsumexp()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::logsumexp(v, 1);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[1, 3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn log_softmax()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::log_softmax(v, 1);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[1, 3]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn softmax_cross_entropy()
{
    let mut graph = ag::Graph::new();
    let ref t = graph.constant(ndarray::arr2(&[[1., 0., 0.]]));
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::softmax_cross_entropy(v, t);
    let ref g = ag::gradients(&[z], &[v], &[None]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn sigmoid_cross_entropy()
{
    let mut graph = ag::Graph::new();
    let ref t = graph.constant(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::sigmoid_cross_entropy(v, t);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[1]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn sparse_softmax_cross_entropy()
{
    let mut graph = ag::Graph::new();
    let ref t = graph.constant(ndarray::arr1(&[1., 0.]));
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::sparse_softmax_cross_entropy(v, t);
    let ref g = ag::gradients(&[z], &[v], &[None]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn gather()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::zeros(&[5, 4, 8, 2]));
    let ref x = graph.constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));
    let ref z = ag::gather(v, x, 2);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[5, 4, 2, 3, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn concat()
{
    let mut graph = ag::Graph::new();
    let ref v1 = graph.variable(ag::ndarray_ext::standard_normal(&[1, 2]));
    let ref v2 = graph.variable(ag::ndarray_ext::standard_normal(&[1, 2]));
    let ref z = ag::concat(&[v1, v2], 1);
    let ref g = ag::gradients(&[z], &[v1], &[Some(&graph.ones(&[1, 4]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v1], graph, 1e-3, 1e-3);
    // FIXME: uncommenting below causes SEGV
    // ag::helper::gradient_check(z, &[v1, v2], g.as_slice(), &ag::Input::new(), 1e-3, 1e-3);
}

#[test]
fn slice()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::slice(v, &[0, 0], &[-1, 2]); // numpy equivalent is v[:, 0:2]
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[4, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn split()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[3, 7, 5]));
    let ref z = ag::split(v, &[2, 3, 2], 1);
    let ref g = ag::gradients(&[&z[1]], &[v], &[Some(&graph.ones(&[3, 3, 5]))]);
    ag::test_helper::gradient_check(&z[1], g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn flatten()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::flatten(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[16]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reshape()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::reshape(v, &[4, 2, 2]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[4, 2, 2]))]);
    ag::test_helper::gradient_check(z, g.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn reshape_grad()
{
    let mut graph = ag::Graph::new();
    let ref v = graph.variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::reshape(&(v * v), &[4, 2, 2]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&graph.ones(&[4, 2, 2]))])[0];
    let ref gg = ag::gradients(&[g], &[v], &[Some(&graph.ones(&[4, 4]))]);
    ag::test_helper::gradient_check(g, gg.as_slice(), &[v], graph, 1e-3, 1e-3);
}

#[test]
fn primitive_back_propagation_through_time()
{
    let mut graph = ag::Graph::new();
    let max_sent = 3;
    let batch_size = 2;

    let ref lookup_table = graph.variable(ag::ndarray_ext::standard_normal(&[5, 3]));
    // (vector_dim -> vocab)
    let ref wo = graph.variable(ag::ndarray_ext::standard_normal(&[3, 5]));
    // (vector_dim -> vector_dim)
    let ref wh = graph.variable(ag::ndarray_ext::standard_normal(&[3, 3]));

    // -- build graph for BPTT --
    let mut loss_buf = vec![];
    let mut h_buf = vec![graph.placeholder()];
    let sentences = graph.placeholder();

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
    graph.feed(
        &sentences,
        ndarray::arr2(&[[2., 3., 1., 3.], [0., 2., 0., 1.]]),
    );
    graph.feed(&h_buf[0], ag::ndarray_ext::zeros(&[batch_size, 3]));
    let params = &[lookup_table, wo, wh];
    let ref g = ag::gradients(&[loss], params, &[Some(&graph.ones(&[batch_size, 1]))]);
    ag::test_helper::gradient_check(loss, g.as_slice(), params, graph, 1e-3, 1e-3);
}

#[test]
pub fn lstm_lm()
{
    let mut graph = ag::Graph::new();
    let state_size = 3;
    let vec_dim = 4;
    let max_sent = 2;
    let vocab_size = 5;
    let batch_size = 2;

    // === graph def
    let ref tbl = graph.variable(ag::ndarray_ext::standard_uniform(&[vocab_size, vec_dim]));
    let ref w = graph.variable(ag::ndarray_ext::standard_normal(&[state_size, vocab_size]));
    let ref sentences = graph.placeholder();
    let ref mut rnn = ag::nn_impl::rnn::LSTM::new(state_size, vec_dim, batch_size, &mut graph);

    let losses = (0..max_sent)
        .map(|i| {
            let ref cur_id = ag::slice(sentences, &[0, i], &[-1, i + 1]);
            let ref nex_id = ag::slice(sentences, &[0, i + 1], &[-1, i + 2]);
            let ref x = ag::gather(tbl, cur_id, 0);
            let ref h = ag::rnn_step(x, rnn, i == max_sent - 1, &mut graph);
            let ref prediction = ag::matmul(h, w);
            ag::sparse_softmax_cross_entropy(prediction, nex_id)
        })
        .collect::<Vec<_>>();

    let loss = losses.last().unwrap();
    let mut vars = rnn.list_vars();
    vars.extend_from_slice(&[tbl, w]);
    let ref g = ag::gradients(&[loss], vars.as_slice(), &[None]);
    graph.feed(sentences, ndarray::arr2(&[[2., 3., 1.], [3., 0., 1.]]));
    ag::test_helper::gradient_check(loss, g.as_slice(), vars.as_slice(), graph, 1e-3, 1e-3);
}
