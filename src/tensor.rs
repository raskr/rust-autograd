extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::cell::Cell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;


/// Symbolic multi-dimensional array.
pub struct Tensor(pub Rc<RawTensor>);

#[doc(hidden)]
/// Symbolic multi-dimensional array.
pub struct RawTensor {
    /// Operation created this node.
    pub op: Box<ops::Op>,

    /// References to immediate predecessors.
    pub inputs: Vec<Tensor>,

    /// Rank number for topological ordering in a graph.
    pub top_rank: usize,

    /// Symbolic shape of this tensor.
    pub shape: Option<Tensor>,

    /// Variable or constant array is placed here.
    pub persistent_array: Option<NdArray>,

    /// Used to lookup a resource of this tensor.
    pub resource_lookup_key: Cell<usize>,

    /// Immutable flag of tensor is placeholder or not.
    pub is_placeholder: bool,
}


impl Tensor {
    #[allow(mutable_transmutes)]
    pub fn get_persistent_array(&self) -> Option<&mut NdArray>
    {
        let m: &mut Option<NdArray> = unsafe { mem::transmute(&self.persistent_array) };
        m.as_mut()
    }


    /// Evaluates this tensor as a ndarray's array object.
    ///
    /// See [eval](../fn.eval.html).
    pub fn eval<'a, 'b: 'a, 'c: 'a, T>(&self, feeds: T) -> NdArray
    where
        T: IntoIterator<Item = &'a (&'b Tensor, &'c ndarray::Array<f32, ndarray::IxDyn>)>,
    {
        ::eval::eval(&[self], feeds).swap_remove(0)
    }


    /// Returns the (symbolic) shape of this tensor.
    ///
    /// See [shape](../ops/fn.shape.html).
    pub fn shape(&self) -> Tensor
    {
        ::ops::shape(self)
    }


    /// Returns the (symbolic) rank of this tensor.
    ///
    /// See [rank](../ops/fn.rank.html).
    pub fn rank(&self) -> Tensor
    {
        ::ops::rank(self)
    }


    /// Returns the (symbolic) size of this tensor.
    ///
    /// See [size](../ops/fn.size.html).
    pub fn size(&self) -> Tensor
    {
        ::ops::size(self)
    }


    #[doc(hidden)]
    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool
    {
        self.inputs.is_empty()
    }
}

impl AsRef<Tensor> for Tensor {
    #[inline(always)]
    fn as_ref(&self) -> &Tensor
    {
        self
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
            "name={}, inputs={:?}",
            self.0.op.name(),
            input_names.as_slice()
        )
    }
}


/// Implementors can be converted to `Tensor`.
pub trait ArrayLike {
    fn as_tensor(&self) -> Tensor;
}

impl ArrayLike for Tensor {
    fn as_tensor(&self) -> Tensor
    {
        self.clone()
    }
}


macro_rules! impl_array_like_for_array {
    ($scalar_type:ty, $num_elems:expr) => {
        impl ArrayLike for [$scalar_type; $num_elems] {
            fn as_tensor(&self) -> Tensor
            {
                    let vec = self
                        .iter()
                        .map(|&a| a as f32 )
                        .collect::<Vec<f32>>();

                    // unwrap is safe
                    let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                    ops::convert_to_tensor(arr)
            }
        }
    };
}


impl_array_like_for_array!(f32, 0);
impl_array_like_for_array!(f32, 1);
impl_array_like_for_array!(f32, 2);
impl_array_like_for_array!(f32, 3);
impl_array_like_for_array!(f32, 4);
impl_array_like_for_array!(f32, 5);
impl_array_like_for_array!(f32, 6);
impl_array_like_for_array!(f32, 7);
impl_array_like_for_array!(f32, 8);

impl_array_like_for_array!(f64, 0);
impl_array_like_for_array!(f64, 1);
impl_array_like_for_array!(f64, 2);
impl_array_like_for_array!(f64, 3);
impl_array_like_for_array!(f64, 4);
impl_array_like_for_array!(f64, 5);
impl_array_like_for_array!(f64, 6);
impl_array_like_for_array!(f64, 7);
impl_array_like_for_array!(f64, 8);

impl_array_like_for_array!(i32, 0);
impl_array_like_for_array!(i32, 1);
impl_array_like_for_array!(i32, 2);
impl_array_like_for_array!(i32, 3);
impl_array_like_for_array!(i32, 4);
impl_array_like_for_array!(i32, 5);
impl_array_like_for_array!(i32, 6);
impl_array_like_for_array!(i32, 7);
impl_array_like_for_array!(i32, 8);

impl_array_like_for_array!(i64, 0);
impl_array_like_for_array!(i64, 1);
impl_array_like_for_array!(i64, 2);
impl_array_like_for_array!(i64, 3);
impl_array_like_for_array!(i64, 4);
impl_array_like_for_array!(i64, 5);
impl_array_like_for_array!(i64, 6);
impl_array_like_for_array!(i64, 7);
impl_array_like_for_array!(i64, 8);

impl_array_like_for_array!(isize, 0);
impl_array_like_for_array!(isize, 1);
impl_array_like_for_array!(isize, 2);
impl_array_like_for_array!(isize, 3);
impl_array_like_for_array!(isize, 4);
impl_array_like_for_array!(isize, 5);
impl_array_like_for_array!(isize, 6);
impl_array_like_for_array!(isize, 7);
impl_array_like_for_array!(isize, 8);

impl_array_like_for_array!(usize, 0);
impl_array_like_for_array!(usize, 1);
impl_array_like_for_array!(usize, 2);
impl_array_like_for_array!(usize, 3);
impl_array_like_for_array!(usize, 4);
impl_array_like_for_array!(usize, 5);
impl_array_like_for_array!(usize, 6);
impl_array_like_for_array!(usize, 7);
impl_array_like_for_array!(usize, 8);

impl_array_like_for_array!(u32, 0);
impl_array_like_for_array!(u32, 1);
impl_array_like_for_array!(u32, 2);
impl_array_like_for_array!(u32, 3);
impl_array_like_for_array!(u32, 4);
impl_array_like_for_array!(u32, 5);
impl_array_like_for_array!(u32, 6);
impl_array_like_for_array!(u32, 7);
impl_array_like_for_array!(u32, 8);

impl_array_like_for_array!(u64, 0);
impl_array_like_for_array!(u64, 1);
impl_array_like_for_array!(u64, 2);
impl_array_like_for_array!(u64, 3);
impl_array_like_for_array!(u64, 4);
impl_array_like_for_array!(u64, 5);
impl_array_like_for_array!(u64, 6);
impl_array_like_for_array!(u64, 7);
impl_array_like_for_array!(u64, 8);
