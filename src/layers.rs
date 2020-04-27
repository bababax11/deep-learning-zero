use ndarray::prelude::*;
use crate::functions::{softmax, cross_entropy_err};

trait Layer {
    fn forward(x: Array2<f64>) -> Array2<f64>;
    fn backward(x: Array2<f64>) -> Array2<f64>;
}

struct AffineLayer {
    W: Array2<f64>,
    b: Array1<f64>,
    x: Option<Array2<f64>>,
    dW: Option<Array2<f64>>,
    db: Option<Array1<f64>>,
}

impl AffineLayer {
    fn new(W: Array2<f64>, b: Array1<f64>) -> Self {
        Self { W: W, b: b , x: None, dW: None, db: None }
    }
    fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.x = Some(x);
        self.x.as_ref().unwrap().dot(&self.W) + &self.b
    }
    fn backward(&mut self, dout: Array2<f64>) -> Array2<f64> {
        let dx = dout.dot(&self.W.t());
        self.dW = Some(self.x.as_ref().unwrap().t().dot(&dout));
        self.db = Some(dout.sum_axis(Axis(0)));
        dx
    }
}

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