extern crate autograd as ag;
extern crate ndarray;

use std::collections::hash_set::HashSet;


// ndarray_extial gradient (ones)
fn ndarray_ext_grad(val: f32, objective_shape: &[usize]) -> ag::Tensor
{
    let arr = ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(objective_shape), val);
    ag::constant(arr)
}

#[test]
fn contributed_to_grads()
{
    // dummy graph
    let ref t = ag::constant(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref z = ag::sigmoid_cross_entropy(&v, &t);
    let booleans = ag::topology::contributed_to_grads(z, &[v]);
    assert_eq!(booleans.len(), 3);
    assert!(!booleans.get(t).unwrap());
    assert!(booleans.get(v).unwrap());
    assert!(booleans.get(z).unwrap());
}

#[test]
fn topological_ordering()
{
    let ref a = ag::constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    let ref v = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref b = ag::variable(ag::ndarray_ext::zeros(&[4, 3]));
    let ref z = ag::matmul(a, v) + b;
    let mut vars = [a, v, b, z];
    // `sort_by_key` don't reverse the order of `a` and `v`
    vars.sort_by_key(|a| a.borrow().rank);
    assert!(vars == [a, v, b, z])
}

#[test]
fn topological_ordering_on_reverse_mode()
{
    let ref x = ag::constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    let ref w = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    let ref b = ag::variable(ag::ndarray_ext::zeros(&[4, 3]));
    let ref z = ag::matmul(x, w) + b;
    let ref g = ag::gradients(z, &[w], Some(&ndarray_ext_grad(1., &[4, 3])))[0];

    let collected = collect_nodes_from(g);
    // to vec
    let mut collected = collected.into_iter().collect::<Vec<ag::Tensor>>();
    // sort by rank
    collected.sort_by_key(|t| t.borrow().rank);
    // tensor to name
    let sorted_names = collected
        .into_iter()
        .map(|t| t.borrow().op.name().to_string())
        .collect::<Vec<String>>();
    // compare
    let boolean = sorted_names ==
        vec![
            "Constant".to_string(), // one or x
            "Constant".to_string(), // one or x
            "SwapAxes".to_string(), // transpose for x
            "MatMul".to_string(),
        ]; // MatMulGrad
    assert!(boolean);
}


// This is used for tests for now
fn collect_nodes_from(end_point: &ag::Tensor) -> HashSet<ag::Tensor>
{
    let mut collected = HashSet::new();
    end_point.visit_once(&mut |arg| { collected.insert(arg.clone()); });
    collected
}
