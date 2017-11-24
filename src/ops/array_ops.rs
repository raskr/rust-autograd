extern crate ndarray;

use Tensor;
use ndarray_ext;
use ndarray_ext::NdArray;
use ops;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::result::Result;


pub struct Broadcast {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

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

pub struct Clip {
    pub min: f32,
    pub max: f32,
}

pub struct ClipGrad {
    pub min: f32,
    pub max: f32,
}

pub struct AddN;

pub struct Gather {
    pub axis: isize,
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


impl ops::Op for Shape {
    fn name(&self) -> &str
    {
        "Shape"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = xs[0];
        Ok(ndarray_ext::shape_of(x))
    }
}

impl ops::Op for Rank {
    fn name(&self) -> &str
    {
        "Rank"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        let x: &NdArray = xs[0];
        Ok(NdArray::from_elem(ndarray::IxDyn(&[]), x.ndim() as f32))
    }
}

impl ops::Op for Size {
    fn name(&self) -> &str
    {
        "Size"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        let x: &NdArray = xs[0];
        Ok(NdArray::from_elem(ndarray::IxDyn(&[]), x.len() as f32))
    }
}


impl ops::Op for Reshape {
    fn name(&self) -> &str
    {
        "Reshape"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let ret = xs[0].clone();
        let shape_arr: &NdArray = xs[1];
        let target = shape_arr
            .iter()
            .map(|&dim_size| if dim_size != -1. {
                dim_size as usize
            } else {
                let product: f32 = shape_arr.iter().product();
                ret.len() / -product as usize
            })
            .collect::<Vec<_>>();

        if let Ok(a) = ret.into_shape(ndarray::IxDyn(target.as_slice())) {
            Ok(a)
        } else {
            Err(::OpComputeErrorStatus::BadInput(
                "Shape incompatible".to_string(),
            ))
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![
            Some(ops::apply_op(Reshape, &[gy, &ops::shape(inputs[0])], None)),
            None,
        ]
    }
}

impl ops::Op for SetDiff1D {
    fn name(&self) -> &str
    {
        "SetDiff1D"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0: &NdArray = xs[0];
        let x1: &NdArray = xs[1];

        let set_a: HashSet<isize> =
            HashSet::from_iter(x0.as_slice().unwrap().iter().map(|&a| a as isize));

        let set_b: HashSet<isize> =
            HashSet::from_iter(x1.as_slice().unwrap().iter().map(|&a| a as isize));

        let diff = set_a.difference(&set_b);

        let mut vec = diff.collect::<Vec<&isize>>();
        vec.sort();
        let vec = vec.into_iter().map(|&a| a as f32).collect::<Vec<f32>>();
        let len = vec.len();
        // safe unwrap
        Ok(
            NdArray::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap(),
        )
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}

impl ops::Op for IndexOp {
    fn name(&self) -> &str
    {
        "IndexOp"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x: &NdArray = xs[0];
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let flat_x = x.view().into_shape((x.len())).unwrap();
        if let Some(ret) = flat_x.get(i) {
            Ok(ndarray::arr0(*ret).into_dyn())
        } else {
            Err(::OpComputeErrorStatus::BadInput(
                "Index out of bounds".to_string(),
            ))
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = IndexOpGrad { index: self.index };
        vec![
            Some(ops::apply_op(op, &[inputs[0], gy], Some(inputs[0].shape()))),
        ]
    }
}

impl ops::Op for IndexOpGrad {
    fn name(&self) -> &str
    {
        "IndexOpGrad"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
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
            .unwrap()  // safe unwrap
            .get_mut(i)
        {
            *a = gy[ndarray::IxDyn(&[])];
        } else {
            return Err(::OpComputeErrorStatus::BadInput(
                "Index out of bounds".to_string(),
            ));
        }
        Ok(result)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Gather {
    fn name(&self) -> &str
    {
        "Gather"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let indices = xs[0].map(|a| *a as usize);
        let param = &xs[1];
        let param_shape = param.shape();
        let axis = if self.axis < 0 {
            (param.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let output_shape: Vec<usize> = {
            let former: &[usize] = &param_shape[..axis];
            let latter: &[usize] = &param_shape[axis + 1..];
            // doing former + indices.shape() + latter
            former
                .into_iter()
                .chain(indices.shape())
                .chain(latter)
                .cloned()
                .collect()
        };

        let flat_indices = indices.into_raw_vec();
        let selected = param.select(ndarray::Axis(axis), flat_indices.as_slice());
        Ok(selected.into_shape(output_shape.as_slice()).unwrap())
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = GatherGrad { axis: self.axis };

        vec![
            None,
            Some(ops::apply_op(
                grad_op,
                &[inputs[0], inputs[1], gy],
                Some(inputs[0].shape()),
            )),
        ]
    }
}


impl ::Op for GatherGrad {
    fn name(&self) -> &str
    {
        "GatherGrad"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let indices: &NdArray = xs[0];
        let param: &NdArray = xs[1];
        let param_shape = param.shape();
        let gy: &NdArray = xs[2];
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
            let i = i as isize;
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
            gx_sliced.zip_mut_with(&gy_sub, |gx, &gy| { *gx += gy; });
        }

        Ok(gx)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None, None]
    }
}

impl ops::Op for AddN {
    fn name(&self) -> &str
    {
        "AddN"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        if 0 == xs.len() {
            unreachable!()
        } else if 1 == xs.len() {
            Err(::OpComputeErrorStatus::Delegate { to: 0 })
        } else if 2 == xs.len() {
            Ok(xs[0] + xs[1])
        } else {
            let mut base = xs[0] + xs[1];
            for &x in xs[2..].iter() {
                base += x;
            }
            Ok(base)
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        (0..inputs.len())
            .map(|_| Some(gy.clone()))
            .collect::<Vec<Option<_>>>()
    }
}

impl ops::Op for Clip {
    fn name(&self) -> &str
    {
        "Clip"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].mapv(move |a| a.min(self.max).max(self.min)))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = ops::apply_op(
            ClipGrad { min: self.min, max: self.max },
            &[inputs[0], gy],
            Some(gy.shape()),
        );
        vec![Some(op)]
    }
}

impl ops::Op for ClipGrad {
    fn name(&self) -> &str
    {
        "ClipGrad"
    }
    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let mut ret = xs[0].mapv(move |x| {
            // x > min && x < max
            (((x > self.min) as i32) as f32) * (((x < self.max) as i32) as f32)
        });
        ret *= xs[1];
        Ok(ret)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}

impl ops::Op for Concat {
    fn name(&self) -> &str
    {
        "Concat"
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        let mut views = vec![];
        for x in xs.iter() {
            views.push(x.view());
        }

        let axis = if self.axis < 0 {
            (xs[0].ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        if let Ok(y) = ndarray::stack(ndarray::Axis(axis), views.as_slice()) {
            Ok(y)
        } else {
            Err(::OpComputeErrorStatus::BadInput(
                "Can't concat arrays whose shapes are incompatible."
                    .to_string(),
            ))
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        // [x1, x2, x3, ..., gy]
        let mut merged_inputs: Vec<&Tensor> = inputs.to_vec();
        merged_inputs.insert(0, gy);
        let merged_inputs: &[&Tensor] = merged_inputs.as_slice();

        let gxs = (0..inputs.len())
            .map(move |i| {
                let grad_op = ConcatGrad { index: i, axis: self.axis };
                Some(ops::apply_op(
                    grad_op,
                    merged_inputs,
                    Some(inputs[0].shape()),
                ))
            })
            .collect::<Vec<Option<Tensor>>>();
        gxs
    }
}

impl ops::Op for ConcatGrad {
    fn name(&self) -> &str
    {
        "ConcatGrad"
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
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
        Ok(gy.slice(&*indices).to_owned())
    }

    fn grad(&self, _: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        (0..inputs.len()).map(|_| None).collect::<Vec<_>>()
    }
}

impl ops::Op for Tile {
    fn name(&self) -> &str
    {
        "Tile"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x: &NdArray = xs[0];

        let axis = if self.axis >= 0 {
            self.axis as usize
        } else {
            x.ndim() - 1
        };

        let mut views = vec![];
        for _ in 0..self.num {
            views.push(x.view());
        }
        if let Ok(ret) = ndarray::stack(ndarray::Axis(axis), views.as_slice()) {
            Ok(ret)
        } else {
            Err(::OpComputeErrorStatus::BadInput(
                "Input shapes incompatible".to_string(),
            ))
        }
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::reduce_sum(gy, &[self.axis], true))]
    }
}

impl ops::Op for Split {
    fn name(&self) -> &str
    {
        "Split"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = xs[0];

        let axis = if self.axis < 0 {
            (x.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let start_index = self.sizes[..self.index].iter().cloned().sum::<usize>() as isize;
        let end_index = start_index + self.sizes[self.index] as isize;
        let indices = make_indices_split(x, start_index, end_index, axis);
        Ok(x.slice(indices.as_slice()).to_owned())
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = SplitGrad {
            axis: self.axis,
            sizes: self.sizes.clone(),
            index: self.index,
        };
        vec![
            Some(ops::apply_op(op, &[inputs[0], gy], Some(inputs[0].shape()))),
        ]
    }
}

impl ops::Op for SplitGrad {
    fn name(&self) -> &str
    {
        "SplitGrad"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
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

        gx.slice_mut(indices.as_slice()).zip_mut_with(
            gy,
            |a, &g| *a = g,
        );
        Ok(gx)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

#[inline]
fn make_indices_split(
    x: &NdArray,
    start_index: isize,
    end_index: isize,
    axis: usize,
) -> Vec<ndarray::Si>
{
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

impl ops::Op for Slice {
    fn name(&self) -> &str
    {
        "Slice"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let y: NdArray = xs[0].slice(&*self.indices).to_owned();
        // TODO: for now, if the size of last axis is 1, removing it.
        let last_axis = y.ndim() - 1;
        let ret = if y.shape()[last_axis] == 1 {
            y.remove_axis(ndarray::Axis(last_axis))
        } else {
            y
        };
        Ok(ret)
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = SliceGrad { indices: self.indices.clone() };
        vec![
            Some(ops::apply_op(op, &[inputs[0], gy], Some(inputs[0].shape()))),
        ]
    }
}

impl ops::Op for SliceGrad {
    fn name(&self) -> &str
    {
        "SliceGrad"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = xs[0];
        let gy = xs[1];
        let mut gx = NdArray::zeros(x.shape());
        // sliced view
        gx.slice_mut(&*self.indices).zip_mut_with(
            &gy,
            |a, &g| *a = g,
        );
        Ok(gx)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        // is this ok?
        vec![None, None]
    }
}
impl ops::Op for Squeeze {
    fn name(&self) -> &str
    {
        "Squeeze"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let mut x = xs[0].view();
        let mut axes = xs[1].iter().map(|&a| a as isize).collect::<Vec<_>>();
        axes.sort();
        let mut adjust = 0;
        for &i in axes.iter() {
            let axis = if i < 0 {
                (x.ndim() as isize + i as isize) as usize
            } else {
                i as usize
            };
            let axis = axis - adjust;
            assert_eq!(1, x.shape()[axis], "Can't squeeze the dim whose size != 1");
            // axis making ok
            x = x.remove_axis(ndarray::Axis(axis));
            adjust += 1;
        }
        Ok(x.to_owned())
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::expand_dims(gy, inputs[1])), None]
    }
}

impl ops::Op for ExpandDims {
    fn name(&self) -> &str
    {
        "ExpandDims"
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        let ret = xs[0].clone();
        let mut axes = xs[1].iter().map(|&a| a as isize).collect::<Vec<_>>();
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
        Ok(ret.into_shape(output_shape).unwrap())
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::squeeze(gy, inputs[1])), None]
    }
}

impl ops::Op for Broadcast {
    fn name(&self) -> &str
    {
        "Broadcast"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = xs[0];
        let target_shape = ndarray_ext::vec_as_shape(xs[1]);
        let axes = xs[2];

        Ok(ndarray_ext::broadcast_to(
            x,
            target_shape.as_slice(),
            axes,
            self.keep_dims,
            self.sparse_axes,
        ))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let sum = ops::reduction_ops::ReduceSum {
            keep_dims: self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let axes = inputs[2];
        let grad_op = ops::apply_op(sum, &[gy, axes], None);
        vec![Some(grad_op), None, None]
    }
}
