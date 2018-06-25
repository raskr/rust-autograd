extern crate ndarray;
extern crate rayon;
use self::rayon::iter::*;

use ::tensor::Tensor;
use super::*;

pub struct MaxPool2D {
    pub pad: usize,
    pub stride: usize,
    pub size: usize
}

pub struct MaxPool2DGrad {
    pad: usize,
    stride: usize,
    size: usize
}

pub struct MaxPool2DGradGrad {
    pad: usize,
    stride: usize,
    size: usize
}

impl ::op::Op for MaxPool2D {
    fn name(&self) -> &str
    {
        "MaxPool"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let x: &NdArray = xs[0];
        let x_shape = x.shape();
        let batch = x_shape[0];
        let c = x_shape[1];
        let xh = x_shape[2];
        let xw = x_shape[3];

        let yh = (xh + 2 * self.pad - self.size) / self.stride + 1;
        let yw = (xw + 2 * self.pad - self.size) / self.stride + 1;
        let all_len_y = batch * c * yh * yw;
        let output = alloc_uninitialized_buf(all_len_y);
        let indices = alloc_uninitialized_buf(all_len_y);
        (0..batch).into_par_iter().for_each(|b| {
            max_pool_unbatched(unsafe { &*x.as_ptr() }, self.pad, xh, xw, yh, yw, c, b,
                               self.size, self.stride,
                               unsafe { &*output.as_ptr() },
                               unsafe { &*indices.as_ptr() }
            );
        });
        let output = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, yh, yw]), output);
        let indices = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, yh, yw]), indices);
        vec![Ok(output.unwrap()), Ok(indices.unwrap())]
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>>
    {
        let indices = ::ops::nth_tensor(y, 1);
        let gx = Tensor::builder()
            .set_inputs(vec![&gy, &indices])
            .build(
                MaxPool2DGrad {
                    pad: self.pad,
                    stride: self.stride,
                    size: self.size
                }
            );
        vec![Some(gx)]
    }
}

#[test]
fn test_max_pool2d()
{
    use ::op::Op;

    let op = MaxPool2D {
        pad: 0,
        stride: 1,
        size: 2
    };
    let x = vec![
        0., 1., 2.,
        5., 4., 3.,
        6., 7., 8.
    ];
    let y = op.compute(::runtime::OpComputeContext {
        node: &::zeros(&[0]),
        xs: vec![&NdArray::from_shape_vec(ndarray::IxDyn(&[1, 1, 3, 3]), x).unwrap()]
    });
    assert_eq!(vec![5., 4., 7., 8.], y[0].as_ref().unwrap().as_slice().unwrap());
    assert_eq!(vec![3., 4., 7., 8.], y[1].as_ref().unwrap().as_slice().unwrap());
}

impl ::op::Op for MaxPool2DGrad {
    fn name(&self) -> &str
    {
        "MaxPoolGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let gy = xs[0];
        let argmax = xs[1];
        let gy_shape = gy.shape();
        let batch = gy_shape[0];
        let c = gy_shape[1];
        let yh = gy_shape[2];
        let yw = gy_shape[3];

        let xh = self.stride * (yh - 1) - 2 * self.pad + self.size;
        let xw = self.stride * (yw - 1) - 2 * self.pad + self.size;
        let gx = vec![0.; batch * c * xh * xw];
        max_pool_grad(unsafe { &*gy.as_ptr() }, yh, yw, c, batch,
                      unsafe { &*gx.as_ptr() },
                      unsafe { &*argmax.as_ptr() });
        let gx = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, xh, xw]), gx);
        vec![Ok(gx.unwrap())]
    }

    fn grad(&self, ggx: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let argmax = xs[1];
        let ggy = Tensor::builder()
            .set_inputs(vec![ggx, argmax])
            .build(MaxPool2DGradGrad {
                pad: self.pad,
                stride: self.stride,
                size: self.size
            });
        vec![Some(ggy), None]
    }
}

impl ::op::Op for MaxPool2DGradGrad {
    fn name(&self) -> &str
    {
        "MaxPoolGradGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let ggx = xs[0];
        let x_shape = ggx.shape();
        let batch= x_shape[0];
        let c = x_shape[1];
        let xh = x_shape[2];
        let xw = x_shape[3];
        let yh = (xh + 2 * self.pad - self.size) / self.stride + 1;
        let yw = (xw + 2 * self.pad - self.size) / self.stride + 1;
        let argmax = xs[1];
        let ggy = alloc_uninitialized_buf(batch * c * yh * yw);
        max_pool_grad_grad(
            unsafe { &*ggx.as_ptr() }, yh, yw, c, batch,
            unsafe { &*ggy.as_ptr() },
            unsafe { &*argmax.as_ptr() }
        );
        let ggy = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, yh, yw]), ggy).unwrap();
        vec![Ok(ggy)]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {

        vec![None, None]
    }
}
