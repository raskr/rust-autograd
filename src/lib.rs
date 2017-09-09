// == module declarations ==
#[macro_use]
#[doc(hidden)]
pub mod test_helper;

pub mod tensor;

#[doc(hidden)]
pub mod topology;

pub mod ops;

pub mod nn_impl;

pub mod sgd;

pub mod graph_sources;

pub mod initializers;

pub mod ndarray_ext;

pub mod train;

pub mod dataset;

// == re-exposures ==
pub use tensor::Tensor;

pub use ops::*;

pub use graph_sources::*;

pub use initializers as init;

pub use tensor::Input;

pub use ndarray_ext::NdArray;
