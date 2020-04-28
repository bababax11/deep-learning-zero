use crate::functions::{cross_entropy_err, softmax};
use ndarray::prelude::*;
use std::ops::{Add, Mul};
trait Layer {
    fn forward(x: Array2<f64>) -> Array2<f64>;
    fn backward(x: Array2<f64>) -> Array2<f64>;
}

#[derive(Clone, Debug, PartialEq)]
struct AffineLayer {
    w: Array2<f64>,
    b: Array1<f64>,
    x: Option<Array2<f64>>,
    dw: Option<Array2<f64>>,
    db: Option<Array1<f64>>,
}

impl AffineLayer {
    fn new(w: Array2<f64>, b: Array1<f64>) -> Self {
        Self {
            w: w,
            b: b,
            x: None,
            dw: None,
            db: None,
        }
    }
    fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.x = Some(x);
        self.x.as_ref().unwrap().dot(&self.w) + &self.b
    }
    fn backward(&mut self, dout: Array2<f64>) -> Array2<f64> {
        let dx = dout.dot(&self.w.t());
        self.dw = Some(self.x.as_ref().unwrap().t().dot(&dout));
        self.db = Some(dout.sum_axis(Axis(0)));
        dx
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SoftMaxWithLossLayer {
    loss: Option<Array1<f64>>,
    y: Option<Array2<f64>>,
    t: Option<Array2<f64>>,
}

impl SoftMaxWithLossLayer {
    fn new() -> Self {
        Self {
            loss: None,
            y: None,
            t: None,
        }
    }
    fn forward(&mut self, x: Array2<f64>, t: Array2<f64>) -> Array1<f64> {
        self.t = Some(t);
        self.y = Some(softmax(&x));
        self.loss = Some(cross_entropy_err(
            self.y.as_ref().unwrap(),
            self.t.as_ref().unwrap(),
        ));
        self.loss.as_ref().unwrap().clone()
    }
    fn backward(&self) -> Array2<f64> {
        let batch_size = self.t.as_ref().unwrap().shape()[0] as f64;
        (self.y.as_ref().unwrap() - self.t.as_ref().unwrap()) / batch_size
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AddLayer<T: Add + Mul + Clone> {
    x: T,
    y: T,
}
impl<T: Add + Mul + Clone> AddLayer<T> {
    fn forward(x: T, y: T) -> <T as Add>::Output {
        x + y
    }
    fn backward(dout: T) -> (T, T) {
        (dout.clone(), dout.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct MulLayer<T: Add + Mul + Clone> {
    x: Option<T>,
    y: Option<T>,
}
impl MulLayer<f64> {
    fn new() -> Self {
        Self { x: None, y: None }
    }
    fn forward(&mut self, x: f64, y: f64) -> f64 {
        self.x = Some(x);
        self.y = Some(y);
        x * y
    }
    fn backward(&self, dout: f64) -> (f64, f64) {
        (dout * self.y.unwrap(), dout * self.x.unwrap())
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::constants::EPS;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn mul() {
        let mut a = MulLayer::new();
        let mut b = MulLayer::new();
        let tmp = a.forward(100.0, 2.0);
        let ans = b.forward(tmp, 1.1);
        assert_approx_eq!(ans, 220.0, EPS);

        let (dap, dt) = b.backward(1.0);
        let (da, dan) = a.backward(dap);
        assert_approx_eq!(da, 2.2, EPS);
        assert_approx_eq!(dan, 110.0, EPS);
        assert_approx_eq!(dt, 200.0, EPS);
    }

    #[test]
    fn add_and_mul() {
        let mut al = MulLayer::new();
        let mut ol = MulLayer::new();
        let mut tl = MulLayer::new();
        let ap = al.forward(100.0, 2.0);
        let op = ol.forward(150.0, 3.0);
        let all_p = AddLayer::forward(ap, op);
        let p = tl.forward(all_p, 1.1);
        assert_approx_eq!(p, 715.0, EPS);

        let (dall_p, dt) = tl.backward(1.0);
        let (dap, dop) = AddLayer::backward(dall_p);
        let (dor, don) = ol.backward(dop);
        let (da, dan) = al.backward(dap);
        assert_approx_eq!(dan, 110.0, EPS);
        assert_approx_eq!(da, 2.2, EPS);
        assert_approx_eq!(dor, 3.3, EPS);
        assert_approx_eq!(don, 165.0, EPS);
        assert_approx_eq!(dt, 650.0, EPS);
    }
}
