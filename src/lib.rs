// == module declarations ==
#[macro_use]
pub mod test_helper;

pub mod tensor;

#[doc(hidden)]
pub mod topology;

pub mod ops;

pub mod nn_impl;

pub mod sgd;

pub mod ndarray_ext;

// == re-exposures ==

pub use ndarray_ext::array_gen as initializers;

pub use initializers as init;

pub use tensor::Tensor;

pub use ops::*;

pub use tensor::Input;

#[doc(hidden)]
pub use ndarray_ext::NdArray;
