extern crate ndarray;
extern crate fnv;

use self::fnv::FnvHashMap;
use context;
use ndarray_ext::NdArray;
use ops;
use std::cmp::Ordering;
use std::collections::hash_set::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;


/// Symbolic multi-dimensional array.
pub struct Tensor(pub Rc<RawTensor>);

pub struct RawTensor {
    /// Operation created this node.
    pub op: Box<ops::Op>,

    /// References to immediate predecessors.
    pub inputs: Vec<Tensor>,

    /// Rank number for topological ordering.
    pub top_rank: usize,
}

/// Implementors can be converted to `Tensor`
pub trait ArrayLike {
    fn as_tensor(&self) -> Tensor;
    fn as_tensor_positive(&self) -> Tensor;
    fn as_reshape_arg_tensor(&self) -> Tensor;
    fn as_axes_tensor(&self) -> Tensor;
}


impl Tensor {
    /// Evaluates this tensor as a ndarray's array object.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let mut g = ag::Graph::new();
    /// let ref x = ag::zeros(&[2, 2]);
    /// assert_eq!(x.eval(&mut g), ndarray::arr2(&[[0., 0.], [0., 0.]]).into_dyn())
    /// ```
    pub fn eval(&self, graph: &mut context::Context) -> NdArray
    {
        graph.eval(&[self]).remove(0)
    }

    #[doc(hidden)]
    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool
    {
        self.inputs.is_empty()
    }

    #[doc(hidden)]
    #[inline]
    pub fn visit_once<F>(&self, f: &mut F)
    where
        F: FnMut(&Tensor) -> (),
    {
        self.run_visit_once(f, &mut HashSet::new())
    }

    #[inline]
    fn run_visit_once<F>(&self, f: &mut F, visited: &mut HashSet<Tensor>)
    where
        F: FnMut(&Tensor) -> (),
    {
        if visited.contains(self) {
            return; // exit early
        } else {
            visited.insert(self.clone()); // first visit
        }

        f(self);

        for child in &(*self).inputs {
            child.run_visit_once(f, visited)
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn visit<F>(&self, f: &mut F)
    where
        F: FnMut(&Tensor) -> (),
    {
        f(self);

        for child in &(*self).inputs {
            child.visit(f)
        }
    }
}

impl Ord for Tensor {
    /// Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering
    {
        self.top_rank.cmp(&other.top_rank)
    }
}

impl PartialOrd for Tensor {
    /// Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
    {
        Some(self.cmp(other))
    }
}

// empty implementation
impl Eq for Tensor {}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool
    {
        // compare addresses on the heap
        Rc::ptr_eq(&self.0, &other.0)
    }
}

// empty implementation
impl Hash for Tensor {
    fn hash<H: Hasher>(&self, _: &mut H)
    {
    }
}

// data is not cloned; only reference count is incremented.
impl Clone for Tensor {
    fn clone(&self) -> Tensor
    {
        Tensor(self.0.clone())
    }
}

impl Deref for Tensor {
    type Target = Rc<RawTensor>;
    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl DerefMut for Tensor {
    fn deref_mut<'a>(&'a mut self) -> &'a mut Rc<RawTensor>
    {
        &mut self.0
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let input_names = self.0
            .inputs
            .iter()
            .map(|a| a.op.name().to_string())
            .collect::<Vec<String>>();
        write!(
            f,
            "op: {}\ninputs: {:?}\n",
            self.0.op.name(),
            input_names.as_slice()
        )
    }
}


#[doc(hidden)]
#[inline]
pub fn eval_tensors(
    tensors: &[Tensor],
    variables: &mut FnvHashMap<Tensor, NdArray>,
    memo: &mut FnvHashMap<Tensor, NdArray>,
) -> Vec<NdArray>
{
    // run graph
    for t in tensors.iter() {
        perform_eval(t, variables, memo, true, 0);
    }

    // extracts target arrays
    let mut evaluated_arrays = Vec::with_capacity(tensors.len());

    for (i, t) in tensors.iter().enumerate() {
        // Need to handle cases where multiple gradient nodes
        // share an output array, and `t` is a variable.
        let contains = tensors[i + 1..].contains(t);
        let in_memo = memo.contains_key(t);
        // Safe unwrapping is guaranteed by ::topology::symbolic_gradients
        match (contains, in_memo) {
            (true, true) => evaluated_arrays.push(memo.get(t).unwrap().clone()),
            (true, false) => evaluated_arrays.push(variables.get(t).unwrap().clone()),
            (false, true) => evaluated_arrays.push(memo.remove(t).unwrap()),
            (false, false) => evaluated_arrays.push(variables.get(t).unwrap().clone()),
        }
    }

    evaluated_arrays
}


#[allow(unused_mut)]
#[doc(hidden)]
#[inline]
/// Performs actual graph traversal and its evaluation
// TODO: loop-based rather than recursion (this would be difficult)
pub fn perform_eval(
    target: &Tensor,
    vars: &mut FnvHashMap<Tensor, NdArray>,
    memo: &mut FnvHashMap<Tensor, NdArray>,
    train: bool,
    mut count: usize, // for debug
)
{
    if vars.contains_key(target) || memo.contains_key(target) {
        return;
    }

    let ref inputs = target.inputs;

    // integrating loops below is impossible because of
    // "memo is already mutably borrowed"
    for x in inputs.iter() {
        perform_eval(x, vars, memo, train, count);
    }

    let y = {
        let mut xs = Vec::with_capacity(inputs.len());
        for x in inputs.iter() {
            if let Some(a) = vars.get(x) {
                // from variable set
                xs.push(a);
            } else {
                // from memo set
                xs.push(memo.get(x).unwrap());
            }
        }
        if target.op.inplace() {
            let mut xs: Vec<&mut NdArray> = unsafe { mem::transmute(xs) };
            target.op.compute_inplace(xs.as_mut_slice(), train);
            None
        } else {
            Some(target.op.compute(xs.as_slice(), train))
        }
    };

    // cache output
    if let Some(a) = y {
        memo.insert(target.clone(), a);
    } else {
        // desired array is always in `memo`
        // because inplace ops don't accept variable/constant.
        let y = memo.remove(&target.inputs[0]);
        // safe unwrap
        memo.insert(target.clone(), y.unwrap());
    }
}


// == ArrayLike impl ==

impl ArrayLike for Tensor {
    fn as_tensor(&self) -> Tensor
    {
        self.clone()
    }

    fn as_tensor_positive(&self) -> Tensor
    {
        self.clone()
    }

    fn as_reshape_arg_tensor(&self) -> Tensor
    {
        self.clone()
    }

    fn as_axes_tensor(&self) -> Tensor
    {
        self.clone()
    }
}


macro_rules! impl_array_like_for_signed_array {
    // placeholder should be `-1`
    ($scalar_type:ty, $placeholder:expr, $num_elems:expr) => {

        impl ArrayLike for [$scalar_type; $num_elems] {
            fn as_tensor(&self) -> Tensor
            {
                let vec = self
                    .iter()
                    .map(|&len| len as f32 )
                    .collect::<Vec<f32>>();

                // unwrap is safe
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                ops::convert_to_tensor(arr)
            }

            fn as_tensor_positive(&self) -> Tensor
            {
                let vec = self
                    .iter()
                    .map(|&axis| {
                        assert!(axis > 0 as $scalar_type, "Non-positive number is not allowed");
                        axis as f32
                     })
                    .collect::<Vec<f32>>();
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                ops::convert_to_tensor(arr)
            }

            fn as_reshape_arg_tensor(&self) -> Tensor
            {
                // validation
                let mut minus_one_found = false;
                let shape = self
                    .iter()
                    .map(|&len| if len == $placeholder {
                        if minus_one_found {
                            panic!("`shape` must not have two or more `-1` dim.");
                        }
                        minus_one_found = true;
                        len as f32
                    } else if len < $placeholder {
                        panic!("`shape` contains invalid dim size: {}", len);
                    } else {
                        len as f32
                    })
                    .collect::<Vec<f32>>();

                // unwrap is safe
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), shape).unwrap();
                ops::convert_to_tensor(arr)
            }

            fn as_axes_tensor(&self) -> Tensor
            {
                let len = self.len() as $scalar_type;

                let vec = self
                    .iter()
                    .map(|&axis|
                        if axis > 0 as $scalar_type {
                            assert!(axis < len, "Wrong axis number"); axis as f32
                        } else {
                            assert!(-axis <= len, "Wrong axis number"); axis as f32
                        }
                    )
                    .collect::<Vec<f32>>();

                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                ops::convert_to_tensor(arr)
            }
        }
    };
}

macro_rules! impl_array_like_for_unsigned_array {


    ($scalar_type:ty, $num_elems:expr) => {

        impl ArrayLike for [$scalar_type; $num_elems] {
            fn as_tensor(&self) -> Tensor
            {
                // unwrap is safe
                let arr = NdArray::from_shape_vec(
                    ndarray::IxDyn(&[self.len()]),
                    self.iter().map(|&a| a as f32).collect::<Vec<f32>>(),
                ).unwrap();

                ops::convert_to_tensor(arr)
            }

            fn as_tensor_positive(&self) -> Tensor
            {
                let vec = self
                    .iter()
                    .map(|&x| { assert_ne!(x, 0, "Zero element is invalid"); x as f32 })
                    .collect::<Vec<f32>>();
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                ops::convert_to_tensor(arr)
            }

            fn as_reshape_arg_tensor(&self) -> Tensor
            {
                self.as_tensor()
            }

            fn as_axes_tensor(&self) -> Tensor
            {
                let len = self.len() as $scalar_type;
                let vec = self
                    .iter()
                    .map(|&axis| { assert!(axis < len, "Wrong axis number"); axis as f32 })
                    .collect::<Vec<f32>>();
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                ops::convert_to_tensor(arr)
            }
        }
    };
}

impl_array_like_for_signed_array!(f32, -1., 1);
impl_array_like_for_signed_array!(f32, -1., 2);
impl_array_like_for_signed_array!(f32, -1., 3);
impl_array_like_for_signed_array!(f32, -1., 4);
impl_array_like_for_signed_array!(f32, -1., 5);
impl_array_like_for_signed_array!(f32, -1., 6);
impl_array_like_for_signed_array!(f32, -1., 7);
impl_array_like_for_signed_array!(f32, -1., 8);

impl_array_like_for_signed_array!(f64, -1., 1);
impl_array_like_for_signed_array!(f64, -1., 2);
impl_array_like_for_signed_array!(f64, -1., 3);
impl_array_like_for_signed_array!(f64, -1., 4);
impl_array_like_for_signed_array!(f64, -1., 5);
impl_array_like_for_signed_array!(f64, -1., 6);
impl_array_like_for_signed_array!(f64, -1., 7);
impl_array_like_for_signed_array!(f64, -1., 8);

impl_array_like_for_signed_array!(i32, -1, 1);
impl_array_like_for_signed_array!(i32, -1, 2);
impl_array_like_for_signed_array!(i32, -1, 3);
impl_array_like_for_signed_array!(i32, -1, 4);
impl_array_like_for_signed_array!(i32, -1, 5);
impl_array_like_for_signed_array!(i32, -1, 6);
impl_array_like_for_signed_array!(i32, -1, 7);
impl_array_like_for_signed_array!(i32, -1, 8);

impl_array_like_for_signed_array!(i64, -1, 1);
impl_array_like_for_signed_array!(i64, -1, 2);
impl_array_like_for_signed_array!(i64, -1, 3);
impl_array_like_for_signed_array!(i64, -1, 4);
impl_array_like_for_signed_array!(i64, -1, 5);
impl_array_like_for_signed_array!(i64, -1, 6);
impl_array_like_for_signed_array!(i64, -1, 7);
impl_array_like_for_signed_array!(i64, -1, 8);

impl_array_like_for_signed_array!(isize, -1, 1);
impl_array_like_for_signed_array!(isize, -1, 2);
impl_array_like_for_signed_array!(isize, -1, 3);
impl_array_like_for_signed_array!(isize, -1, 4);
impl_array_like_for_signed_array!(isize, -1, 5);
impl_array_like_for_signed_array!(isize, -1, 6);
impl_array_like_for_signed_array!(isize, -1, 7);
impl_array_like_for_signed_array!(isize, -1, 8);

impl_array_like_for_unsigned_array!(usize, 1);
impl_array_like_for_unsigned_array!(usize, 2);
impl_array_like_for_unsigned_array!(usize, 3);
impl_array_like_for_unsigned_array!(usize, 4);
impl_array_like_for_unsigned_array!(usize, 5);
impl_array_like_for_unsigned_array!(usize, 6);
impl_array_like_for_unsigned_array!(usize, 7);
impl_array_like_for_unsigned_array!(usize, 8);

impl_array_like_for_unsigned_array!(u32, 1);
impl_array_like_for_unsigned_array!(u32, 2);
impl_array_like_for_unsigned_array!(u32, 3);
impl_array_like_for_unsigned_array!(u32, 4);
impl_array_like_for_unsigned_array!(u32, 5);
impl_array_like_for_unsigned_array!(u32, 6);
impl_array_like_for_unsigned_array!(u32, 7);
impl_array_like_for_unsigned_array!(u32, 8);

impl_array_like_for_unsigned_array!(u64, 1);
impl_array_like_for_unsigned_array!(u64, 2);
impl_array_like_for_unsigned_array!(u64, 3);
impl_array_like_for_unsigned_array!(u64, 4);
impl_array_like_for_unsigned_array!(u64, 5);
impl_array_like_for_unsigned_array!(u64, 6);
impl_array_like_for_unsigned_array!(u64, 7);
impl_array_like_for_unsigned_array!(u64, 8);
