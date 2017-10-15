extern crate ndarray;
extern crate fnv;

use self::fnv::FnvHashMap;
use graph;
use ndarray_ext::NdArray;
use ops;
use std::cmp::Ordering;
use std::collections::hash_set::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;


/// Symbolic multi-dimensional array which supports
/// efficient gradient computation.
pub struct Tensor(pub Rc<RawTensor>);

pub struct RawTensor {
    /// Operation created this node
    pub op: Box<ops::Op>,

    /// References to immediate predecessors.
    pub inputs: Vec<Tensor>,

    /// Rank number for topological ordering
    pub top_rank: usize,
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
    ///
    /// let mut g = ag::Graph::new();
    ///
    /// let ref x = g.constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    /// let ref w = g.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    /// let ref b = g.variable(ag::ndarray_ext::zeros(&[1, 3]));
    /// let ref z = ag::matmul(x, w) + b;
    ///
    /// assert_eq!(z.eval(&mut g).shape(), &[4, 3])
    /// ```
    pub fn eval(&self, graph: &mut graph::Graph) -> NdArray
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
        ::topology::perform_eval(t, variables, memo, true, 0);
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


// == ArrayType and its impl ==

pub trait ArrayType {
    fn as_tensor(&self) -> Tensor;
}

impl ArrayType for Tensor {
    fn as_tensor(&self) -> Tensor
    {
        self.clone()
    }
}

impl<'a> ArrayType for &'a Tensor {
    fn as_tensor(&self) -> Tensor
    {
        (*self).clone()
    }
}

macro_rules! impl_unsigned_slice_to_shape {
    ($scalar_type:ty) => {
        impl<'a> ArrayType for &'a [$scalar_type] {
            fn as_tensor(&self) -> Tensor
            {
                // unwrap is safe
                let arr = NdArray::from_shape_vec(
                    ndarray::IxDyn(&[self.len()]),
                    self.iter().map(|&a| a as f32).collect::<Vec<f32>>(),
                ).unwrap();

                ops::convert_to_tensor(arr)
            }
        }
    };
}

macro_rules! impl_signed_slice_to_shape {
    ($scalar_type:ty, $placeholder:expr) => {
        impl<'a> ArrayType for &'a [$scalar_type] {
            fn as_tensor(&self) -> Tensor
            {
                // validation
                let mut minus_one_found = false;
                let shape = self
                    .iter()
                    .map(|&len| if len == $placeholder {
                        if minus_one_found {
                            panic!("`shape` has two or more `-1` dim.");
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
        }
    };
}

macro_rules! impl_signed_array_to_shape {
    ($scalar_type:ty, $placeholder:expr, $num_elems:expr) => {
        impl ArrayType for [$scalar_type; $num_elems] {
            fn as_tensor(&self) -> Tensor
            {
                // validation
                let mut minus_one_found = false;
                let shape = self
                    .iter()
                    .map(|&len| if len == $placeholder {
                        if minus_one_found {
                            panic!("`shape` has two or more `-1` dim.");
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
        }
    };
}

macro_rules! impl_unsigned_array_to_shape {
    ($scalar_type:ty, $num_elems:expr) => {
        impl ArrayType for [$scalar_type; $num_elems] {
            fn as_tensor(&self) -> Tensor
            {
                // unwrap is safe
                let arr = NdArray::from_shape_vec(
                    ndarray::IxDyn(&[self.len()]),
                    self.iter().map(|&a| a as f32).collect::<Vec<f32>>(),
                ).unwrap();

                ops::convert_to_tensor(arr)
            }
        }
    };
}

impl_unsigned_slice_to_shape!(usize);
impl_unsigned_slice_to_shape!(u32);
impl_unsigned_slice_to_shape!(u64);

impl_signed_slice_to_shape!(isize, -1);
impl_signed_slice_to_shape!(i32, -1);
impl_signed_slice_to_shape!(i64, -1);
impl_signed_slice_to_shape!(f32, -1.);
impl_signed_slice_to_shape!(f64, -1.);

// --- array ---
impl_signed_array_to_shape!(f32, -1., 1);
impl_signed_array_to_shape!(f32, -1., 2);
impl_signed_array_to_shape!(f32, -1., 3);
impl_signed_array_to_shape!(f32, -1., 4);
impl_signed_array_to_shape!(f32, -1., 5);
impl_signed_array_to_shape!(f32, -1., 6);

impl_signed_array_to_shape!(f64, -1., 1);
impl_signed_array_to_shape!(f64, -1., 2);
impl_signed_array_to_shape!(f64, -1., 3);
impl_signed_array_to_shape!(f64, -1., 4);
impl_signed_array_to_shape!(f64, -1., 5);
impl_signed_array_to_shape!(f64, -1., 6);

impl_signed_array_to_shape!(i32, -1, 1);
impl_signed_array_to_shape!(i32, -1, 2);
impl_signed_array_to_shape!(i32, -1, 3);
impl_signed_array_to_shape!(i32, -1, 4);
impl_signed_array_to_shape!(i32, -1, 5);
impl_signed_array_to_shape!(i32, -1, 6);

impl_signed_array_to_shape!(i64, -1, 1);
impl_signed_array_to_shape!(i64, -1, 2);
impl_signed_array_to_shape!(i64, -1, 3);
impl_signed_array_to_shape!(i64, -1, 4);
impl_signed_array_to_shape!(i64, -1, 5);
impl_signed_array_to_shape!(i64, -1, 6);

impl_signed_array_to_shape!(isize, -1, 1);
impl_signed_array_to_shape!(isize, -1, 2);
impl_signed_array_to_shape!(isize, -1, 3);
impl_signed_array_to_shape!(isize, -1, 4);
impl_signed_array_to_shape!(isize, -1, 5);
impl_signed_array_to_shape!(isize, -1, 6);

impl_unsigned_array_to_shape!(usize, 1);
impl_unsigned_array_to_shape!(usize, 2);
impl_unsigned_array_to_shape!(usize, 3);
impl_unsigned_array_to_shape!(usize, 4);
impl_unsigned_array_to_shape!(usize, 5);
impl_unsigned_array_to_shape!(usize, 6);

impl_unsigned_array_to_shape!(u32, 1);
impl_unsigned_array_to_shape!(u32, 2);
impl_unsigned_array_to_shape!(u32, 3);
impl_unsigned_array_to_shape!(u32, 4);
impl_unsigned_array_to_shape!(u32, 5);
impl_unsigned_array_to_shape!(u32, 6);

impl_unsigned_array_to_shape!(u64, 1);
impl_unsigned_array_to_shape!(u64, 2);
impl_unsigned_array_to_shape!(u64, 3);
impl_unsigned_array_to_shape!(u64, 4);
impl_unsigned_array_to_shape!(u64, 5);
impl_unsigned_array_to_shape!(u64, 6);
