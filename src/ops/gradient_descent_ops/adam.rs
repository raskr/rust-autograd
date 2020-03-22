use crate::optimizers::adam::StaticParams;
use crate::Float;

pub(crate) struct AdamOp<F: Float> {
    pub(crate) static_params: StaticParams<F>,
}

impl<F: Float> crate::op::Op<F> for AdamOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) {
        let StaticParams { alpha, eps, b1, b2 } = self.static_params;
        let mut t = ctx.input_mut(4);
        let input1 = ctx.input(1);

        // Make new m
        let new_m = {
            let tmp = F::one() - b1;
            let mut input2 = ctx.input_mut(2);
            input2.zip_mut_with(&input1, move |x2_elem, &g| {
                *x2_elem = *x2_elem * b1 + tmp * g
            });
            input2
        };

        // Make new v
        let new_v = {
            let tmp = F::one() - b2;
            let mut input3 = ctx.input_mut(3);
            input3.zip_mut_with(&input1, move |x3_elem, &g| {
                *x3_elem = *x3_elem * b2 + tmp * g * g
            });
            input3
        };

        // Make hat
        let m_hat = {
            let t_val = unsafe { *t.as_ptr() };
            let rhs = F::one() / (F::one() - b2.powf(t_val));
            let v_hat = new_v.mapv(move |new_v_elem| new_v_elem * rhs);
            let rhs = F::one() / (F::one() - b1.powf(t_val));
            let mut m_hat = new_m.mapv(move |new_m_elem| new_m_elem * rhs);
            m_hat.zip_mut_with(&v_hat, move |a, &b| (*a) /= b.sqrt() + eps);
            m_hat
        };

        // Update t and variable
        ctx.input_mut(0)
            .zip_mut_with(&m_hat, move |l, &r| *l -= alpha * r);
        unsafe {
            *t.as_mut_ptr() += F::one();
        }

        ctx.append_empty_output();
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        ctx.append_input_grad(None);
    }
}
