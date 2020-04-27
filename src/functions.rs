use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use crate::constants::EPS;

pub fn softmax(arr: &Array2<f64>) -> Array2<f64> {
    let max = arr.max().unwrap();
    let exp_arr = (arr-*max).mapv_into(f64::exp);
    let sum_exp = exp_arr.sum_axis(Axis(0));
    exp_arr / sum_exp
}

pub fn sigmoid(arr: &Array2<f64>) -> Array2<f64> {
    arr.mapv(|x| 1.0 / (f64::exp(-x) + 1.0))
}

pub fn cross_entropy_err(y: &Array2<f64>, t: &Array2<f64>) -> Array2<f64> {
    - (t * &(y+EPS).mapv_into(f64::ln))
}
