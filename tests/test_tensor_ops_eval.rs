extern crate autograd as ag;
extern crate ndarray;

#[test]
fn reduce_prod()
{
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[3, 2]));
    let ref z = ag::reduce_prod(v, &[0, 1], false); // keep_dims=false
    let ref x = ag::equal(z, v);
    let ref k = z * v;
    z.eval(&[])[ndarray::IxDyn(&[])];
}
