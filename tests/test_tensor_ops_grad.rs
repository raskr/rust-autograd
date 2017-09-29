extern crate autograd as ag;
extern crate ndarray;


// ones
fn init_grad(val: f32, objective_shape: &[usize]) -> ag::Tensor
{
    let arr = ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(objective_shape), val);
    ag::constant(arr)
}

#[test]
fn get()
{
    let ref v = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref a: ag::Tensor = 2 * v;
    let ref z = a.get(1);
    let ref g = ag::gradients(&[z], &[v], &[None]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn add_n()
{
    let ref v1 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v2 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref v3 = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref z = ag::add_n(&[v1, v2, v3]);
    let ref g = ag::gradients(&[z], &[v2], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v2], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn clip()
{
    let ref v = ag::variable(ndarray::arr1(&[1., 2., 3.]));
    let ref z = ag::clip(v, 1.5, 2.5);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
    assert_eq!(g[0].eval(), ndarray::arr1(&[0., 1., 0.]).into_dyn())
}

#[test]
fn asinh()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::asinh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn acosh()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::acosh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn atanh()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::atanh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn sinh()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::sinh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn cosh()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::cosh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn tanh()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::tanh(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn asin()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::asin(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-2);
}

#[test]
fn acos()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::acos(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn atan()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::atan(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn sin()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::sin(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn cos()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::cos(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn tan()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::tan(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-2);
}

#[test]
fn pow()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0., 0.2));
    let ref z = ag::pow(v, 1.2);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn exp()
{
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 0.9, 1.1));
    let ref z = ag::exp(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-2);
}

#[test]
fn log()
{
    use std::f32;
    let ref v = ag::variable(ag::ndarray_ext::random_uniform(&[3], 1., 1.1));
    let ref z = ag::log(v, f32::consts::E);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-2);
}

#[test]
fn expand_dims()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3]));
    let ref z = ag::expand_dims(v, &[0, 2]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[1, 3, 1]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn squeeze()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 1, 2, 1]));
    let ref z = ag::squeeze(v, &[3, 1]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn matmul()
{
    let ref a = ag::constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::matmul(a, v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[4, 3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn batch_matmul()
{
    let ref a = ag::constant(ag::ndarray_ext::standard_normal(&[2, 4, 2]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2, 3]));
    let ref z = ag::batch_matmul(a, v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[2, 4, 3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn wx_plus_b()
{
    let ref a = ag::constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref b = ag::variable(ag::ndarray_ext::zeros(&[4, 3]));
    let ref z = ag::matmul(a, v) + b;
    let ref g = ag::gradients(&[z], &[b], &[Some(&init_grad(1., &[4, 3]))]);
    ag::test_helper::gradient_check(z, &[b], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_min()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_min(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_min_keep()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_min(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_max()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_max(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_max_keep()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_max(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_mean()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_mean(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_mean_keep()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_mean(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_sum()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_sum(v, 1, false); // keep_dims=false
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reduce_sum_keep()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2, 2]));
    let ref z = ag::reduce_sum(v, 1, true); // keep_dims=true
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[3, 1, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reverse_axes()
{
    let ref v = ag::constant(ag::ndarray_ext::zeros(&[2, 3, 4, 5]));
    let ref z = ag::reverse_axes(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[5, 4, 3, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn transpose()
{
    let ref v = ag::constant(ag::ndarray_ext::zeros(&[1, 2, 3, 4, 5]));
    let ref z = ag::transpose(v, &[4, 2, 3, 0, 1]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[5, 3, 4, 1, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reshape_after_transpose()
{
    let ref v = ag::constant(ag::ndarray_ext::zeros(&[1, 2, 3, 4, 5]));
    let ref z = ag::transpose(v, &[4, 2, 3, 0, 1]);
    let ref z = ag::reshape(z, &[15, 8]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[15, 8]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn add()
{
    let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = a + b;
    let ref g = ag::gradients(&[z], &[a, b], &[Some(&init_grad(1., &[2, 2]))]);
    ag::test_helper::gradient_check(z, &[a, b], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn mul()
{
    let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = a * b;
    let ref g = ag::gradients(&[z], &[a, b], &[Some(&init_grad(1., &[2, 2]))]);
    ag::test_helper::gradient_check(z, &[a, b], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn sigmoid()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::sigmoid(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[2, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn elu()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::elu(v, 1.);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[2, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn relu()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 2]));
    let ref z = ag::relu(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[2, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn logsumexp()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::logsumexp(v, 1);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[1, 3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn log_softmax()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::log_softmax(v, 1);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[1, 3]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn softmax_cross_entropy()
{
    let ref t = ag::constant(ndarray::arr2(&[[1., 0., 0.]]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::softmax_cross_entropy(v, t);
    let ref g = ag::gradients(&[z], &[v], &[None]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn sigmoid_cross_entropy()
{
    let ref t = ag::constant(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[1, 3]));
    let ref z = ag::sigmoid_cross_entropy(v, t);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[1]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn sparse_softmax_cross_entropy()
{
    let ref t = ag::constant(ndarray::arr1(&[1., 0.]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::sparse_softmax_cross_entropy(v, t);
    let ref g = ag::gradients(&[z], &[v], &[None]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn gather()
{
    let ref v = ag::variable(ag::ndarray_ext::zeros(&[5, 4, 8, 2]));
    let ref x = ag::constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));
    let ref z = ag::gather(v, x, 2);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[5, 4, 2, 3, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn concat()
{
    let ref v1 = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2]));
    let ref v2 = ag::variable(ag::ndarray_ext::standard_normal(&[1, 2]));
    let ref z = ag::concat(&[v1, v2], 1);
    let ref g = ag::gradients(&[z], &[v1], &[Some(&init_grad(1., &[1, 4]))]);
    ag::test_helper::gradient_check(z, &[v1], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
    // FIXME: uncommenting below causes SEGV
    // ag::helper::gradient_check(z, &[v1, v2], g.as_slice(), &ag::Input::new(), 1e-3, 1e-3);
}

#[test]
fn slice()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::slice(v, &[0, 0], &[-1, 2]); // numpy equivalent is v[:, 0:2]
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[4, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn split()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 7, 5]));
    let ref z = ag::split(v, &[2, 3, 2], 1);
    let ref g = ag::gradients(&[&z[1]], &[v], &[Some(&init_grad(1., &[3, 3, 5]))]);
    ag::test_helper::gradient_check(&z[1], &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn flatten()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::flatten(v);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[16]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reshape()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::reshape(v, &[4, 2, 2]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[4, 2, 2]))]);
    ag::test_helper::gradient_check(z, &[v], g.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn reshape_grad()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[4, 4]));
    let ref z = ag::reshape(&(v * v), &[4, 2, 2]);
    let ref g = ag::gradients(&[z], &[v], &[Some(&init_grad(1., &[4, 2, 2]))])[0];
    let ref gg = ag::gradients(&[g], &[v], &[Some(&init_grad(1., &[4, 4]))]);
    ag::test_helper::gradient_check(g, &[v], gg.as_slice(), &ag::Feed::new(), 1e-3, 1e-3);
}

#[test]
fn primitive_back_propagation_through_time()
{
    let max_sent = 3;
    let batch_size = 2;

    let ref lookup_table = ag::variable(ag::ndarray_ext::standard_normal(&[5, 3]));
    // (vector_dim -> vocab)
    let ref wo = ag::variable(ag::ndarray_ext::standard_normal(&[3, 5]));
    // (vector_dim -> vector_dim)
    let ref wh = ag::variable(ag::ndarray_ext::standard_normal(&[3, 3]));

    // -- build graph for BPTT --
    let mut loss_buf = vec![];
    let mut h_buf = vec![ag::placeholder()];
    let sentences = ag::placeholder();

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

    // -- graph building end --

    // inputs (batch_size=2, sentence_len=4)
    let ref fd = ag::Feed::new()
        .add(&h_buf[0], ag::ndarray_ext::zeros(&[batch_size, 3]))
        .add(
            &sentences,
            ndarray::arr2(&[[2., 3., 1., 3.], [0., 2., 0., 1.]]),
        );

    let params = &[lookup_table, wo, wh];
    let ref g = ag::gradients(&[loss], params, &[Some(&init_grad(1., &[batch_size, 1]))]);
    ag::test_helper::gradient_check(loss, params, g.as_slice(), fd, 1e-3, 1e-3);
}

#[test]
pub fn lstm_lm()
{
    let state_size = 3;
    let vec_dim = 4;
    let max_sent = 2;
    let vocab_size = 5;
    let batch_size = 2;

    // === graph def
    let ref tbl = ag::variable(ag::ndarray_ext::standard_normal(&[vocab_size, vec_dim]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[state_size, vocab_size]));
    let sentences = ag::placeholder();
    let mut rnn = ag::nn_impl::rnn::LSTM::new(state_size, vec_dim, batch_size);

    let mut loss_buf = vec![];
    for i in 0..max_sent {
        let cur_id = ag::slice(&sentences, &[0, i], &[-1, i + 1]);
        let nex_id = ag::slice(&sentences, &[0, i + 1], &[-1, i + 2]);
        let x = ag::gather(tbl, &cur_id, 0);
        let h = ag::rnn_step(&x, &mut rnn, i == max_sent - 1);
        let prediction = ag::matmul(&h, w);
        loss_buf.push(ag::sparse_softmax_cross_entropy(&prediction, &nex_id));
    }
    let loss = loss_buf.last().unwrap();
    // === graph def end

    // == graph building end ==
    let ref fd = ag::Feed::new().add(&sentences, ndarray::arr2(&[[2., 3., 1.], [3., 0., 1.]]));

    // ==  test ==
    let mut vars = rnn.list_vars();
    vars.extend_from_slice(&[tbl, w]);

    let ref g = ag::gradients(&[loss], vars.as_slice(), &[None]);
    ag::test_helper::gradient_check(loss, vars.as_slice(), g.as_slice(), fd, 1e-3, 1e-3);
}
