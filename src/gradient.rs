use ops;
use std::cmp::Ordering;
use std::collections::hash_map::{Entry, HashMap};
use std::mem;
use tensor::Tensor;


fn get_sorted_between_nodes<'a>(ys: &[&'a Tensor], xs: &[&'a Tensor]) -> Vec<TensorWrapper<'a>>
{
    let marked = mark_backward_path(ys, xs);
    let mut between = marked
        .into_iter()
        .filter_map(|(k, v)| if v && !k.is_source() { Some(k.wrapped()) } else { None })
        .collect::<Vec<TensorWrapper>>();
    between.sort_unstable();  // topological sort
    between
}

#[test]
fn test_sorted_between_nodes()
{
    use std::rc::Rc;
    // dummy graph
    // y = 3 * x1 * x1 + 5 * x2 + x3;
    let ref x1 = ::placeholder(&[]);
    let ref x2 = ::placeholder(&[]);
    let ref x3 = ::placeholder(&[]);
    let ref a = 3 * x1;  // rank 1
    let ref b = a * x1;  // rank 2
    let ref c = 5 * x2;  // rank 1
    let ref d = b + c;   // rank 3
    let ref y = d + x3;  // rank 4
    let sorted_between: Vec<TensorWrapper> = get_sorted_between_nodes(&[y], &[x1, x2]);
    let sorted_between: Vec<&Tensor> = sorted_between.iter().map(|a| a.tensor).collect();

    assert!(sorted_between.contains(&a));
    assert!(sorted_between.contains(&b));
    assert!(sorted_between.contains(&c));
    assert!(sorted_between.contains(&d));
    assert!(sorted_between.contains(&y));
    assert_eq!(sorted_between.len(), 5);

    // Order test
    assert!(sorted_between[..2].contains(&a));
    assert!(sorted_between[..2].contains(&c));
    assert!(Rc::ptr_eq(sorted_between[2], b));
    assert!(Rc::ptr_eq(sorted_between[3], d));
    assert!(Rc::ptr_eq(sorted_between[4], y));
}

use std::fmt;

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.op.name())
    }
}

/// Returns symbolic gradient tensors.
///
/// This computes partial derivatives of `ys` with `xs` and returns the
/// gradients. This is achieved by building the subgraph between `ys` and
/// `xs` in reverse order from user's graph definition.
/// `known_gys` are already known gradients of `ys`'s outputs.
///
/// NOTE: Nodes that do not contribute to the gradient won't be included to avoid
/// unnecessary computation.
pub fn symbolic_gradients(ys: &[&Tensor], xs: &[&Tensor],
                          known_gys: &[Option<&Tensor>]) -> Vec<Tensor>
{
    assert_eq!(ys.len(), known_gys.len(), "`ys.len()` must match `gys.len()`");

    // Prepare gradient store
    let mut computed_gys: HashMap<&Tensor, Vec<Tensor>> = HashMap::new();

    // Store default grads
    for (y, known_gy) in ys.iter().zip(known_gys) {
        let gy = known_gy.map(|k| k.clone()).unwrap_or_else(|| ops::scalar(1.));
        computed_gys.insert(y, vec![gy]);
    }

    // "Reverse order" iteration
    for y in get_sorted_between_nodes(ys, xs).iter().rev() {
        // Get y's input nodes
        let xs_ref = y.tensor.inputs.iter().map(|a| a).collect::<Vec<_>>();

        // Get gxs by calling `Op::grad`
        let gxs = {
            let gys = computed_gys.get_mut(&y.tensor).expect(&format!(
                "Couldn't get `{}`'s grad (probably a bug)", y.tensor));
            accumulate_grads(gys);
            y.tensor.op.grad(&gys[0], xs_ref.as_slice(), y.tensor)
        };

        // Check correctness of y's `grad` implementation
        debug_assert_eq!(xs_ref.len(), gxs.len(), "`{}.grad` must return {} gxs",
                         y.tensor, xs_ref.len());

        // Register computed gxs
        for (gx, x) in gxs.into_iter().zip(xs_ref) {
            if let Some(gx_) = gx {
                // Store gx in `computed_gys[x]`
                match computed_gys.entry(x) {
                    Entry::Occupied(mut grad_buf) => grad_buf.get_mut().push(gx_),
                    Entry::Vacant(grad_buf) => {
                        grad_buf.insert(vec![gx_]);
                    }
                }
            }
        }
    }

    // Aggregate and return xs's gradients
    xs.iter()
        .map(|x| {
            let mut gxs = computed_gys.remove(x).expect(
                "Input tensor(s) not differentiable.",
            );
            accumulate_grads(&mut gxs);
            gxs.remove(0)
        })
        .collect::<Vec<Tensor>>()
}

fn mark_backward_path<'a>(ys: &[&'a Tensor], xs: &[&'a Tensor]) -> HashMap<&'a Tensor, bool>
{
    fn rec<'a>(target: &'a Tensor, vars: &[&'a Tensor], memo: &mut HashMap<&'a Tensor, bool>)
    {
        if memo.contains_key(target) {
            return;
        }

        let mut contrib = false;

        if !target.op.stop_gradient() {
            if vars.contains(&target) {
                contrib = true; // need not recursion.
            } else {
                for x in target.inputs.iter() {
                    // recurse
                    rec(x, vars, memo);
                    // unwrap is always safe
                    contrib |= *memo.get(x).unwrap();
                }
            }
        }
        memo.insert(target, contrib);
    }

    let mut memo = HashMap::new();
    for y in ys.into_iter() {
        rec(y, xs, &mut memo);
    }
    memo
}

impl<'a> Ord for TensorWrapper<'a> {
    /// Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering
    {
        self.tensor.top_rank.cmp(&other.tensor.top_rank)
    }
}

impl<'a> PartialOrd for TensorWrapper<'a> {
    /// Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
    {
        Some(self.tensor.top_rank.cmp(&other.tensor.top_rank))
    }
}

impl<'a> Eq for TensorWrapper<'a> {}

impl<'a> PartialEq for TensorWrapper<'a> {
    fn eq(&self, other: &TensorWrapper<'a>) -> bool
    {
        self.tensor.eq(&other.tensor)
    }
}

impl Tensor {
    #[inline]
    fn wrapped(&self) -> TensorWrapper
    {
        let ret = TensorWrapper {
            tensor: self,
        };
        ret
    }
}

#[inline]
fn accumulate_grads(grads: &mut Vec<Tensor>)
{
    if grads.len() >= 2 {
        let acc = {
            let refs = grads.iter().map(|a| a).collect::<Vec<_>>();
            ops::add_n(refs.as_slice())
        };
        mem::swap(&mut vec![acc], grads);
    }
}

/// Module private.
struct TensorWrapper<'a> {
    tensor: &'a Tensor,
}
