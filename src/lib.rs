// == module declarations ==
#[macro_use]
pub mod test_helper;

pub mod tensor;

#[doc(hidden)]
pub mod topology;

pub mod ops;

pub mod nn_impl;

pub mod sgd;

pub mod initializers;

#[doc(hidden)]
pub mod ndarray_ext;

// == re-exposures ==

pub use tensor::Tensor;

pub use ops::*;

pub use initializers as init;

pub use tensor::Input;

pub use ndarray_ext::NdArray;
