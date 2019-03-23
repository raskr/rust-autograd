use ndarray;
use ndarray_ext;
use ndarray_ext::NdArray;
use op;
use ops;
use std::collections::HashSet;
use std::iter::FromIterator;
use tensor::Tensor;
use Float;

pub struct ExpandDims;

pub struct Squeeze;

pub struct Slice {
    pub indices: Box<[ndarray::Si]>,
}

pub struct SliceGrad {
    pub indices: Box<[ndarray::Si]>,
}

pub struct Split {
    pub axis: isize,
    pub sizes: Vec<usize>,
    pub index: usize,
}

pub struct SplitGrad {
    pub axis: isize,
    pub sizes: Vec<usize>,
    pub index: usize,
}

pub struct Tile {
    pub axis: isize,
    pub num: usize,
}

pub struct Concat {
    pub axis: isize,
}

pub struct ConcatGrad {
    pub axis: isize,
    pub index: usize,
}

pub struct Clip<T: Float> {
    pub min: T,
    pub max: T,
}

pub struct ClipGrad<T: Float> {
    pub min: T,
    pub max: T,
}

pub struct AddN;

pub struct Gather {
    pub axis: isize,
    pub should_normalize_negative_indices: bool,
}

pub struct GatherGrad {
    pub axis: isize,
}

pub struct IndexOp {
    pub index: isize,
}

pub struct IndexOpGrad {
    pub index: isize,
}

pub struct SetDiff1D;

pub struct Shape;

pub struct Rank;

pub struct Size;

pub struct Reshape;

pub struct InferBinOpShape;

impl<T: Float> op::Op<T> for InferBinOpShape {
    fn name(&self) -> &str {
        "InferBinOpShape"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let a_shape_float = xs[0];
        let b_shape_float = xs[1];
        let a_shape = a_shape_float
            .map(|x| x.clone().to_usize().unwrap())
            .into_raw_vec();
        let b_shape = b_shape_float
            .map(|x| x.clone().to_usize().unwrap())
            .into_raw_vec();
        let a_is_scalar = ndarray_ext::is_scalar_shape(a_shape.as_slice());
        let b_is_scalar = ndarray_ext::is_scalar_shape(b_shape.as_slice());

        let ret = if !a_is_scalar && !b_is_scalar {
            let a_rank = a_shape.len();
            let b_rank = b_shape.len();
            assert_eq!(a_rank, b_rank);
            let max = a_shape
                .iter()
                .zip(b_shape)
                .map(|(a, b)| T::from(a.clone().max(b.clone())).unwrap())
                .collect::<Vec<T>>();
            Ok(NdArray::from_shape_vec(ndarray::IxDyn(&[a_rank]), max).unwrap())
        } else if !a_is_scalar {
            Err(::op::ComputeException::Delegate { to: 0 })
        } else {
            Err(::op::ComputeException::Delegate { to: 1 })
        };
        vec![ret]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None]
    }
}

impl<T: Float> op::Op<T> for Shape {
    fn name(&self) -> &str {
        "Shape"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        vec![Ok(ndarray_ext::shape_of(x))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Rank {
    fn name(&self) -> &str {
        "Rank"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x: &NdArray<T> = xs[0];
        vec![Ok(NdArray::from_elem(
            ndarray::IxDyn(&[]),
            T::from(x.ndim()).unwrap(),
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Size {
    fn name(&self) -> &str {
        "Size"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x: &NdArray<T> = xs[0];
        vec![Ok(NdArray::from_elem(
            ndarray::IxDyn(&[]),
            T::from(x.len()).unwrap(),
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let ret = xs[0].clone();
        let shape_arr: &NdArray<T> = xs[1];
        let target = shape_arr
            .iter()
            .map(|&dim_size| {
                if dim_size != -T::one() {
                    dim_size.to_usize().unwrap()
                } else {
                    let product: T = shape_arr.iter().fold(T::one(), |acc, &x| acc * x);
                    ret.len() / product.neg().to_usize().unwrap()
                }
            })
            .collect::<Vec<_>>();

        let ret = if let Ok(a) = ret.into_shape(ndarray::IxDyn(target.as_slice())) {
            Ok(a)
        } else {
            panic!("Shape Incompatible");
        };
        vec![ret]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_inputs(vec![gy, &ops::shape(inputs[0])])
            .build(Reshape);
        vec![Some(gx), None]
    }
}

impl<T: Float> op::Op<T> for SetDiff1D {
    fn name(&self) -> &str {
        "SetDiff1D"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x0: &NdArray<T> = xs[0];
        let x1: &NdArray<T> = xs[1];

        let set_a: HashSet<isize> = HashSet::from_iter(
            x0.as_slice()
                .unwrap()
                .iter()
                .map(|&a| a.to_isize().unwrap()),
        );

        let set_b: HashSet<isize> = HashSet::from_iter(
            x1.as_slice()
                .unwrap()
                .iter()
                .map(|&a| a.to_isize().unwrap()),
        );

        let diff = set_a.difference(&set_b);

        let mut vec = diff.collect::<Vec<&isize>>();
        vec.sort();
        let vec = vec
            .into_iter()
            .map(|&a| T::from(a).unwrap())
            .collect::<Vec<T>>();
        let len = vec.len();
        // safe unwrap
        let ret = Ok(NdArray::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap());
        vec![ret]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None]
    }
}

impl<T: Float> op::Op<T> for IndexOp {
    fn name(&self) -> &str {
        "IndexOp"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x: &NdArray<T> = xs[0];
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let flat_x = x.view().into_shape(x.len()).unwrap();
        let ret = if let Some(ret) = flat_x.get(i) {
            Ok(ndarray::arr0(*ret).into_dyn())
        } else {
            panic!("Index out of bounds");
        };
        vec![ret]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let op = IndexOpGrad { index: self.index };
        let gx = Tensor::builder()
            .set_shape(inputs[0].shape())
            .set_inputs(vec![inputs[0], gy])
            .build(op);
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for IndexOpGrad {
    fn name(&self) -> &str {
        "IndexOpGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let gy = xs[1];
        let mut result = NdArray::zeros(x.shape());
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let len = result.len();
        if let Some(a) = result
            .view_mut()
            .into_shape(len)
            .unwrap() // safe unwrap
            .get_mut(i)
        {
            *a = gy[ndarray::IxDyn(&[])];
        } else {
            panic!("Index out of bounds");
        }
        vec![Ok(result)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Gather {
    fn name(&self) -> &str {
        "Gather"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let param = xs[1];
        let indices = xs[0];
        let indices_shape = indices.shape();
        let param_shape = param.shape();
        let axis = ndarray_ext::normalize_negative_axis(self.axis, param.ndim());

        let output_shape: Vec<usize> = {
            let former: &[usize] = &param_shape[..axis];
            let latter: &[usize] = &param_shape[axis + 1..];
            // doing former + indices.shape() + latter
            former
                .into_iter()
                .chain(indices_shape)
                .chain(latter)
                .cloned()
                .collect()
        };

        let flat_indices = if self.should_normalize_negative_indices {
            ndarray_ext::normalize_negative_axes(indices, param_shape[axis])
        } else {
            indices.mapv(|a| a.to_usize().unwrap()).into_raw_vec()
        };
        let selected = param.select(ndarray::Axis(axis), flat_indices.as_slice());
        vec![Ok(selected.into_shape(output_shape.as_slice()).unwrap())]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_shape(inputs[0].shape())
            .set_inputs(vec![inputs[0], inputs[1], gy])
            .build(GatherGrad { axis: self.axis });
        vec![None, Some(gx)]
    }
}

impl<T: Float> op::Op<T> for GatherGrad {
    fn name(&self) -> &str {
        "GatherGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let indices: &NdArray<T> = xs[0];
        let param: &NdArray<T> = xs[1];
        let param_shape = param.shape();
        let gy: &NdArray<T> = xs[2];
        let axis = if self.axis == -1 {
            param.ndim()
        } else {
            self.axis as usize
        };

        // get read-only view of gy and reshape it
        let gy = {
            let former = &param_shape[..axis];
            let latter = &param_shape[axis + 1..];
            let shape: Vec<usize> = former
                .into_iter()
                .chain(&[indices.len()])
                .chain(latter)
                .cloned()
                .collect();
            gy.view().into_shape(shape).unwrap()
        };

        let mut gx = NdArray::zeros(param.shape());

        for (gy_sub, &i) in gy.axis_iter(ndarray::Axis(axis)).zip(indices) {
            let i = i.to_isize().unwrap();
            // get gx's sub view
            let gx_sliced = gx.slice_mut(
                (0..param.ndim())
                    .map(|dim| {
                        if dim == axis {
                            ndarray::Si(i, Some(i + 1), 1) // squeezed later
                        } else {
                            ndarray::Si(0, None, 1)
                        }
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
            );

            // squeeze
            let mut gx_sliced = gx_sliced.remove_axis(ndarray::Axis(axis));
            // assign gy to sliced view
            gx_sliced.zip_mut_with(&gy_sub, |gx, &gy| {
                *gx += gy;
            });
        }

        vec![Ok(gx)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None, None]
    }
}

impl<T: Float> op::Op<T> for AddN {
    fn name(&self) -> &str {
        "AddN"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let ret = if 0 == xs.len() {
            unreachable!()
        } else if 1 == xs.len() {
            Err(::op::ComputeException::Delegate { to: 0 })
        } else if 2 == xs.len() {
            Ok(xs[0] + xs[1])
        } else {
            let mut base = xs[0] + xs[1];
            for &x in xs[2..].iter() {
                base += x;
            }
            Ok(base)
        };
        vec![ret]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        (0..inputs.len())
            .map(|_| Some(gy.clone()))
            .collect::<Vec<Option<_>>>()
    }
}

impl<T: Float> op::Op<T> for Clip<T> {
    fn name(&self) -> &str {
        "Clip"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].mapv(move |a| a.min(self.max).max(self.min)))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_shape(gy.shape())
            .set_inputs(vec![inputs[0], gy])
            .build(ClipGrad {
                min: self.min,
                max: self.max,
            });
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for ClipGrad<T> {
    fn name(&self) -> &str {
        "ClipGrad"
    }
    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let mut ret = xs[0].mapv(move |x| {
            // x > min && x < max
            T::from((((x > self.min) as i32) as f32) * (((x < self.max) as i32) as f32)).unwrap()
        });
        ret *= xs[1];
        vec![Ok(ret)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None]
    }
}

impl<T: Float> op::Op<T> for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let mut views = vec![];
        let xs = ctx.grab_inputs();
        for x in xs.iter() {
            views.push(x.view());
        }

        let axis = if self.axis < 0 {
            (xs[0].ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let ret = if let Ok(y) = ndarray::stack(ndarray::Axis(axis), views.as_slice()) {
            Ok(y)
        } else {
            panic!("Can't concat arrays whose shapes are incompatible.");
        };
        vec![ret]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        // [x1, x2, x3, ..., gy]
        let mut merged_inputs: Vec<&Tensor<T>> = inputs.to_vec();
        merged_inputs.insert(0, gy);
        let merged_inputs = merged_inputs.as_slice();

        let gxs = (0..inputs.len())
            .map(move |i| {
                let gx = Tensor::builder()
                    .set_shape(inputs[0].shape())
                    .set_inputs_slice(merged_inputs)
                    .build(ConcatGrad {
                        index: i,
                        axis: self.axis,
                    });
                Some(gx)
            })
            .collect::<Vec<Option<Tensor<T>>>>();
        gxs
    }
}

impl<T: Float> op::Op<T> for ConcatGrad {
    fn name(&self) -> &str {
        "ConcatGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let gy = xs[0];
        let xs = xs[1..].to_vec();

        let axis = if self.axis < 0 {
            (xs[0].ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        // make slice indices
        let mut start_idx = 0;
        for x in xs[..self.index].iter() {
            start_idx += x.shape()[axis];
        }
        let region_len = xs[self.index].shape()[axis] as isize;
        let indices = (0..gy.ndim())
            .map(move |_axis| {
                if _axis == axis {
                    // partial region
                    ndarray::Si(start_idx as isize, Some(region_len), 1)
                } else {
                    // full slice
                    ndarray::Si(0, None, 1)
                }
            })
            .collect::<Vec<ndarray::Si>>();

        // do slice
        vec![Ok(gy.slice(&*indices).to_owned())]
    }

    fn grad(&self, _: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        (0..inputs.len()).map(|_| None).collect::<Vec<_>>()
    }
}

impl<T: Float> op::Op<T> for Tile {
    fn name(&self) -> &str {
        "Tile"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x: &NdArray<T> = xs[0];

        let axis = if self.axis >= 0 {
            self.axis as usize
        } else {
            x.ndim() - 1
        };

        let mut views = vec![];
        for _ in 0..self.num {
            views.push(x.view());
        }
        let ret = if let Ok(ret) = ndarray::stack(ndarray::Axis(axis), views.as_slice()) {
            Ok(ret)
        } else {
            panic!("Shape Incompatible");
        };
        vec![ret]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::reduce_sum(gy, &[self.axis], true))]
    }
}

impl<T: Float> op::Op<T> for Split {
    fn name(&self) -> &str {
        "Split"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];

        let axis = if self.axis < 0 {
            (x.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let start_index = self.sizes[..self.index].iter().cloned().sum::<usize>() as isize;
        let end_index = start_index + self.sizes[self.index] as isize;
        let indices = make_indices_split(x, start_index, end_index, axis);
        vec![Ok(x.slice(indices.as_slice()).to_owned())]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let op = SplitGrad {
            axis: self.axis,
            sizes: self.sizes.clone(),
            index: self.index,
        };
        let gx = Tensor::builder()
            .set_inputs(vec![inputs[0], gy])
            .set_shape(inputs[0].shape())
            .build(op);
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for SplitGrad {
    fn name(&self) -> &str {
        "SplitGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let gy = xs[1];
        let mut gx = NdArray::zeros(x.shape());

        let axis = if self.axis < 0 {
            (x.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let start_index = self.sizes[..self.index].iter().cloned().sum::<usize>() as isize;
        let end_index = start_index + self.sizes[self.index] as isize;
        let indices = make_indices_split(x, start_index, end_index, axis);

        gx.slice_mut(indices.as_slice())
            .zip_mut_with(gy, |a, &g| *a = g);
        vec![Ok(gx)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

#[inline]
fn make_indices_split<T: Float>(
    x: &NdArray<T>,
    start_index: isize,
    end_index: isize,
    axis: usize,
) -> Vec<ndarray::Si> {
    let ndim = x.ndim();
    assert!(ndim > axis, "Wrong split axis");
    (0..ndim)
        .map(|i| {
            if i == axis {
                ndarray::Si(start_index, Some(end_index), 1)
            } else {
                // full slice
                ndarray::Si(0, None, 1)
            }
        })
        .collect::<Vec<ndarray::Si>>()
}

impl<T: Float> op::Op<T> for Slice {
    fn name(&self) -> &str {
        "Slice"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let y: NdArray<T> = xs[0].slice(&*self.indices).to_owned();
        // TODO: for now, if the size of last axis is 1, removing it.
        let last_axis = y.ndim() - 1;
        let ret = if y.shape()[last_axis] == 1 {
            y.remove_axis(ndarray::Axis(last_axis))
        } else {
            y
        };
        vec![Ok(ret)]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let op = SliceGrad {
            indices: self.indices.clone(),
        };
        let gx = Tensor::builder()
            .set_inputs(vec![inputs[0], gy])
            .set_shape(inputs[0].shape())
            .build(op);
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for SliceGrad {
    fn name(&self) -> &str {
        "SliceGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let gy = xs[1];
        let mut gx = NdArray::zeros(x.shape());
        // sliced view
        gx.slice_mut(&*self.indices)
            .zip_mut_with(&gy, |a, &g| *a = g);
        vec![Ok(gx)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        // is this ok?
        vec![None, None]
    }
}
impl<T: Float> op::Op<T> for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let mut x = xs[0].view();
        let mut axes = xs[1]
            .iter()
            .map(|&a| a.to_isize().unwrap())
            .collect::<Vec<_>>();
        axes.sort();
        let mut adjust = 0;
        for &i in axes.iter() {
            let axis = if i < 0 {
                (x.ndim() as isize + i as isize) as usize
            } else {
                i as usize
            };
            let axis = axis - adjust;
            assert_eq!(1, x.shape()[axis], "Can't squeeze a dim whose size != 1");
            // axis making ok
            x = x.remove_axis(ndarray::Axis(axis));
            adjust += 1;
        }
        vec![Ok(x.to_owned())]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::expand_dims(gy, inputs[1])), None]
    }
}

impl<T: Float> op::Op<T> for ExpandDims {
    fn name(&self) -> &str {
        "ExpandDims"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let ret = xs[0].clone();
        let mut axes = xs[1]
            .iter()
            .map(|&a| a.to_isize().unwrap())
            .collect::<Vec<_>>();
        axes.sort();
        let mut output_shape = ret.shape().to_vec();
        for &i in axes.iter() {
            let axis = if i < 0 {
                (ret.ndim() as isize + i as isize) as usize
            } else {
                i as usize
            };
            output_shape.insert(axis, 1);
        }
        vec![Ok(ret.into_shape(output_shape).unwrap())]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::squeeze(gy, inputs[1])), None]
    }
}
