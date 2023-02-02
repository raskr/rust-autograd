use crate::Float;

pub(crate) struct AdaGradOp<F: Float> {
    pub(crate) lr: F,
}

impl<F: Float> crate::op::Op<F> for AdaGradOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), crate::op::OpError> {
        let param = ctx.input_mut(0);
        let grad = ctx.input(1);
        let mut h = ctx.input_mut(2);
        h.zip_mut_with(&grad, |h, &g| *h += g * g);
        let eps = F::from(1e-7).unwrap();
        ndarray::Zip::from(param)
            .and(grad)
            .and(h)
            .for_each(move |p, &g, h| *p -= self.lr * g / (h.sqrt() + eps));

        ctx.append_empty_output();
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}
