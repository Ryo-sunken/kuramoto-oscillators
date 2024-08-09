use matrix::Matrix;
use ode_solver::{EulerSolver, RungeKuttaSolver};
use crate::parameter::{NetworkParam, ControlParam};
use std::f64::consts::PI;
use std::fs::File;

pub struct KuramotoOscillators {
    inc: Matrix<f64>,
    wgt: Matrix<f64>,
    input_wgt: Matrix<f64>,
    av_wgt: Matrix<f64>,
    omega: Matrix<f64>,
    input_omega: Matrix<f64>,
    control_type: usize,
}

impl KuramotoOscillators {
    pub fn new(network_param: &NetworkParam, control_param: &ControlParam) -> Self {
        let edges = network_param
            .connectivity
            .iter()
            .flatten()
            .filter(|&x| *x != 0.)
            .count()
            / 2;
        let mut inc = Matrix::<f64>::zero(network_param.state_dim, edges);
        let mut wgt = Matrix::<f64>::zero(edges, edges);
        let mut av_wgt = Matrix::<f64>::zero(network_param.state_dim, network_param.state_dim);

        let mut edge_count = 0;
        for i in 0..network_param.state_dim {
            for j in i..network_param.state_dim {
                if network_param.connectivity[i][j] != 0. {
                    inc[i][edge_count] = 1.;
                    inc[j][edge_count] = -1.;
                    wgt[edge_count][edge_count] = network_param.connectivity[i][j];
                    edge_count += 1;
                }
            }
        }

        let mut start = 0;
        for k in 0..network_param.cluster_nodes_num.len() {
            let sum: f64 = control_param.average_weight[k].iter().sum();
            for i in 0..network_param.cluster_nodes_num[k] {
                for j in 0..network_param.cluster_nodes_num[k] {
                    av_wgt[i + start][j + start] = control_param.average_weight[k][j] / sum;
                }
            }
            start += network_param.cluster_nodes_num[k];
        }

        Self {
            inc,
            wgt,
            input_wgt: Matrix::<f64>::from_vec_col(control_param.input_weight.clone()).diag(),
            av_wgt,
            omega: Matrix::<f64>::from_vec_col(network_param.frequency.clone()),
            input_omega: Matrix::<f64>::from_vec_col(control_param.input_frequency.clone()),
            control_type: control_param.control_type,
        }
    }
}

impl EulerSolver<f64, File> for KuramotoOscillators {
    fn dot_x(&self, x: &Matrix<f64>, t: f64) -> Matrix<f64> {
        if self.control_type == 0 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.av_wgt * x - x).sin()
        } else if self.control_type == 1 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.input_omega * t - x).sin()
        } else {
            x.clone()
        }
    }

    fn post_process(&self, x: &Matrix<f64>) -> Matrix<f64> {
        x.repeat(0., 2. * PI)
    }
}

impl RungeKuttaSolver<f64, File> for KuramotoOscillators {
    fn dot_x(&self, x: &Matrix<f64>, t: f64) -> Matrix<f64> {
        if self.control_type == 0 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.av_wgt * x - x).sin()
        } else if self.control_type == 1 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.input_omega * t - x).sin()
        } else {
            x.clone()
        }
    }

    fn post_process(&self, x: &Matrix<f64>) -> Matrix<f64> {
        x.repeat(0., 2. * PI)
    }
}