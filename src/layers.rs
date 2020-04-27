use ndarray::prelude::*;
use crate::functions::{softmax, cross_entropy_err};
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
        Self { w: w, b: b , x: None, dw: None, db: None }
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
        Self { loss: None, y: None, t: None}
    }
    fn forward(&mut self, x: Array2<f64>, t: Array2<f64>) -> Array1<f64> {
        self.t = Some(t);
        self.y = Some(softmax(&x));
        self.loss = Some(cross_entropy_err(
            self.y.as_ref().unwrap(), self.t.as_ref().unwrap()
        ));
        self.loss.as_ref().unwrap().clone()
    }
    fn backward(&self) -> Array2<f64> {
        let batch_size = self.t.as_ref().unwrap().shape()[0] as f64;
        (self.y.as_ref().unwrap() - self.t.as_ref().unwrap()) / batch_size
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AddLayer<T: Add+Mul+Clone> {
    x: T,
    y: T,
}
impl<T: Add+Mul+Clone> AddLayer<T> {
    fn forward(x: T, y: T) -> <T as Add>::Output {
        x + y
    }
    fn backward(dout: T) -> (T, T) {
        (dout.clone(), dout.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct MulLayer<T: Add+Mul+Clone> {
    x: Option<T>,
    y: Option<T>,
}
impl MulLayer<f64> {
    fn forward(&mut self, x: f64, y: f64) -> f64 {
        self.x = Some(x);
        self.y = Some(y);
        x * y
    }
    fn backward(&self, dout: f64) -> (f64, f64) {
        (dout * self.y.unwrap(), dout * self.x.unwrap())
    }
}