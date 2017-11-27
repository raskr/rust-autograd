use ops;
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
use std::mem;
use tensor::Tensor;


/// Returns symbolic gradient tensors.
///
/// This computes partial derivatives of `ys` with `xs` and returns the
/// gradients. This is achieved by building the subgraph between `ys` and
/// `xs` in reverse order from user's graph definition.
/// `gys` are already known gradients of `ys`'s outputs.
///
/// NOTE: Nodes that do not contribute to the gradient won't be included to avoid
/// unnecessary computation.
pub fn symbolic_gradients(ys: &[&Tensor], xs: &[&Tensor], gys: &[Option<&Tensor>]) -> Vec<Tensor>
{
    assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");
    #[inline]
    fn maybe_accumulate_grads(gys_: &mut Vec<Tensor>)
    {
        if gys_.len() >= 2 {
            let acc = {
                // values to refs
                let refs = gys_.iter().map(|a| a).collect::<Vec<_>>();
                ops::add_n(refs.as_slice())
            };
            mem::swap(&mut vec![acc], gys_);
        }
    }

    // Mapping of {y => [gy]}
    let mut grads: HashMap<Tensor, Vec<Tensor>> = HashMap::new();

    // Mapping of {node => must visit or not (boolean)}
    let contrib = contributed_to_grads(ys, xs);

    // Prepare a heap with tensor's rank numbers for reverse
    // topological sort.
    let mut heap = BinaryHeap::new();
    for (&o, out_grad) in ys.into_iter().zip(gys) {
        let g = out_grad.map(|gy| gy.clone()).unwrap_or_else(
            || ops::scalar(1.),
        );
        heap.push(o.clone());
        grads.insert(o.clone(), vec![g]);
    }

    // This prevents calling `grad()` twice or more
    let mut grad_done = HashSet::<Tensor>::new();

    // builds backward graph
    while let Some(target) = heap.pop() {
        // Vec<Tensor> to Vec<&Tensor>
        let xs = target.inputs.iter().map(|a| a).collect::<Vec<_>>();

        // time to call `grad`
        let gxs = {
            if let Some(gys) = grads.get_mut(&target) {
                maybe_accumulate_grads(gys);
                target.op.grad(&gys[0], xs.as_slice(), &target)
            } else {
                unreachable!("Safe unwrapping should be guaranteed by topological ordering")
            }
        };

        debug_assert_eq!(
            xs.len(),
            gxs.len(),
            "Wrong `grad` implementation of {}. Must return {} gxs.",
            target,
            xs.len()
        );

        // register computed gxs
        for (x, maybe_gx) in xs.into_iter().zip(gxs) {
            if !contrib.contains_key(x) {
                continue;
            }
            // cuts the backward path if gx is None.
            if let Some(gx) = maybe_gx {
                // memo gx
                if let Some(mut gys) = grads.remove(x) {
                    // gradient accumulation should be delayed here
                    gys.push(gx);
                    grads.insert(x.clone(), gys);
                } else {
                    grads.insert(x.clone(), vec![gx]);
                }
                // update heap
                if !x.is_source() && !grad_done.contains(x) {
                    grad_done.insert(x.clone());
                    heap.push(x.clone());
                }
            }
        }
    }

    xs.iter()
        .map(|v| {
            let mut gys = grads.remove(v).expect(
                "Input tensor(s) didn't contributed to gradient computation",
            );
            maybe_accumulate_grads(&mut gys);
            gys.remove(0)
        })
        .collect::<Vec<Tensor>>()
}


#[inline]
/// Makes mapping of {node => the node contributed to gradient or not}
fn contributed_to_grads(objectives: &[&Tensor], variables: &[&Tensor]) -> HashMap<Tensor, bool>
{
    fn rec(target: &Tensor, vars: &[&Tensor], memo: &mut HashMap<Tensor, bool>)
    {
        if memo.contains_key(target) {
            return;
        }

        let mut contrib = false;

        if vars.contains(&target) {
            contrib = true;
        } else {
            for input in target.inputs.iter() {
                // recurse
                rec(input, vars, memo);
                // unwrap is always safe
                contrib |= *memo.get(input).unwrap();
            }
        }

        memo.insert(target.clone(), contrib);
    }

    let mut memo = HashMap::new();
    for o in objectives.into_iter() {
        rec(o, variables, &mut memo);
    }
    memo
}


#[test]
fn contributed_to_grads_test()
{
    use ndarray_ext;
    // dummy graph
    let mut ctx = ::Context::new();
    let ref t = ::constant(ndarray_ext::standard_normal(&[2, 3]), &mut ctx);
    let ref v = ::variable(ndarray_ext::standard_normal(&[2, 3]), &mut ctx);
    let ref z = ::sigmoid_cross_entropy(&v, &t);
    let booleans = contributed_to_grads(&[z], &[v]);
    assert_eq!(booleans.len(), 3);
    assert!(!booleans.get(t).unwrap());
    assert!(booleans.get(v).unwrap());
    assert!(booleans.get(z).unwrap());
}
